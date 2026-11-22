/**
 * Contract (FORK-RPC-CONTRACT-20261122): topic suggestions in the Inbox.
 *
 * - A suggestion is one visible `topic-suggestion` entry naming user/assistant
 *   text messages of the Inbox's current branch; it creates nothing. Other ids
 *   (tool output, another branch, unknown) are refused: `invalid_message_ids`.
 * - State is folded from append-only `topic-suggestion-state` entries, the
 *   latest winning: dismissed → accepted is allowed, accepted → dismissed is
 *   refused (`already_accepted`), repeats are idempotent successes, unknown ids
 *   are `suggestion_not_found`.
 * - Accept seeds the topic with all of the suggestion's own messages (the
 *   agent's pick; clients never pick), creates the topic and returns before
 *   its seed is ready; the first turn gets the newest 30 of those messages
 *   verbatim (text only), the older ones in one small-model summary
 *   (or a "not summarised" line when that fails) and the reason. The Inbox gets
 *   a visible `topic-continued` note. `topics_new` seeds the same way.
 * - `topics_list` lists open suggestions only; the Inbox board lists them and
 *   the latest dismissed titles; every create/accept/dismiss signals a change.
 * - Organise queues one Inbox turn and refuses another while it is pending.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import type { AssistantMessage } from "@oh-my-pi/pi-ai";
import { createMockModel, registerMockApi } from "@oh-my-pi/pi-ai/providers/mock";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import type { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { InternalUrlRouter } from "@oh-my-pi/pi-coding-agent/internal-urls";
import { IrcBus } from "@oh-my-pi/pi-coding-agent/irc/bus";
import { runRpcTopicsCommand } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-topics";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import type { CustomEntry, CustomMessageEntry } from "@oh-my-pi/pi-coding-agent/session/session-entries";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { ExecutorOptions } from "@oh-my-pi/pi-coding-agent/task/executor";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { formatTopicBoard } from "@oh-my-pi/pi-coding-agent/topics/board";
import { createTopicsHost, type TopicChange, type TopicsHost } from "@oh-my-pi/pi-coding-agent/topics/host";
import { InboxBinding } from "@oh-my-pi/pi-coding-agent/topics/binding";
import type { InboxMessage, TopicsInbox } from "@oh-my-pi/pi-coding-agent/topics/inbox";
import { buildTopicSeed, type SeedSummarizer, smolSeedSummarizer } from "@oh-my-pi/pi-coding-agent/topics/seed";
import {
	foldSuggestions,
	TOPIC_CONTINUED_TYPE,
	TOPIC_ORGANISE_TYPE,
	TOPIC_SUGGESTION_STATE_TYPE,
	TOPIC_SUGGESTION_TYPE,
	TopicsError,
} from "@oh-my-pi/pi-coding-agent/topics/suggestions";
import { resetTopicsRegistryForTests, topicsRegistry } from "@oh-my-pi/pi-coding-agent/topics/workers";
import { VibeSessionRegistry } from "@oh-my-pi/pi-coding-agent/vibe/runtime";
import type { SingleResult } from "@oh-my-pi/pi-tui/tools/task";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createAssistantMessage } from "../helpers/agent-session-setup";

const OWNER = "Main";

function resultFor(options: ExecutorOptions): SingleResult {
	return {
		index: options.index,
		id: options.id,
		agent: options.agent.name,
		agentSource: options.agent.source,
		task: options.task,
		exitCode: 0,
		output: `answer from ${options.id}`,
		stderr: "",
		truncated: false,
		durationMs: 1,
		tokens: 0,
		requests: 0,
	};
}

/** The code a call refuses with. */
async function codeOf(run: () => unknown): Promise<string | undefined> {
	try {
		await run();
	} catch (error) {
		if (error instanceof TopicsError) return error.code;
		throw error;
	}
	throw new Error("expected a refusal");
}

/** One macrotask turn: rejections still unowned after it are reported as unhandled. */
function macrotask(): Promise<void> {
	const { promise, resolve } = Promise.withResolvers<void>();
	setImmediate(resolve);
	return promise;
}

describe("topic suggestions", () => {
	let tempDir: TempDir;
	let manager: AsyncJobManager;
	let sm: SessionManager;
	/** Executor options of every topic's first turn, in spawn order. */
	let firstTurns: ExecutorOptions[];
	let hosts: TopicsHost[];
	let queuedTurns: InboxMessage[];
	let turnPending: boolean;
	/** The open Inbox session. */
	let inboxSm: SessionManager;
	/** Counts the sessions the Inbox adopted, as the real Inbox does: a reopened session is a new transcript. */
	let inboxEpoch: number;
	/** Awaited by every state write before it saves; a rejection fails the write. */
	let stateWrite: (() => Promise<void>) | undefined;

	/** Open `next` in the Inbox: a `/new`, a fork, or a reopen of an earlier session. */
	function switchInbox(next: SessionManager): void {
		inboxSm = next;
		inboxEpoch++;
	}

	function parent(): ToolSession {
		return {
			cwd: tempDir.path(),
			settings: Settings.isolated(),
			asyncJobManager: manager,
			sessionManager: sm,
			getAgentId: () => OWNER,
			getSessionId: () => sm.getSessionId(),
			getSessionFile: () => sm.getSessionFile() ?? null,
			getArtifactsDir: () => null,
			taskDepth: 0,
			enableLsp: false,
		} as unknown as ToolSession;
	}

	/**
	 * The Inbox over `inboxSm` (`sm` unless a test switches sessions): notes are saved at once (an idle Inbox);
	 * a state write awaits `stateWrite` first and, like the real one, adds nothing when it fails.
	 */
	function inbox(): TopicsInbox {
		const identity = () => ({
			sessionId: inboxSm.getSessionId(),
			sessionFile: inboxSm.getSessionFile(),
			sessionEpoch: inboxEpoch,
			transcriptEpoch: 0,
		});
		return {
			bind: () => new InboxBinding(identity),
			branch: binding => {
				binding.assertCurrent();
				return inboxSm.getBranch();
			},
			entries: binding => {
				binding.assertCurrent();
				return inboxSm.getEntriesView();
			},
			appendState: async (binding, customType, data, check) => {
				binding.assertCurrent();
				await stateWrite?.();
				binding.assertCurrent();
				check?.();
				inboxSm.appendCustomEntry(customType, data);
			},
			appendNote: async (binding, { customType, content, details }) => {
				if (!binding.isCurrent) return false;
				inboxSm.appendCustomMessageEntry(customType, content, true, details, "agent");
				return true;
			},
			queueTurn: async (binding, _customType, render) => {
				binding.assertCurrent();
				queuedTurns.push(render());
				turnPending = true;
				return { queued: false };
			},
			turnPending: () => turnPending,
		};
	}

	function hostWith(summarize: SeedSummarizer = async () => "SUMMARY OF EARLIER"): TopicsHost {
		const host = createTopicsHost(parent(), { inbox: inbox(), summarize });
		hosts.push(host);
		return host;
	}

	/** Run every topic's first turn the way the executor leaves it (child transcript, idle ref). */
	function mockFirstTurns(): void {
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async options => {
			firstTurns.push(options);
			const childFile = path.join(
				path.dirname(sm.getSessionFile()!),
				path.basename(sm.getSessionFile()!, ".jsonl"),
				`${options.id}.jsonl`,
			);
			fs.mkdirSync(path.dirname(childFile), { recursive: true });
			const header = { type: "session", version: 3, id: `${options.id}-s`, timestamp: "", cwd: tempDir.path() };
			const init = { type: "session_init", id: "i1", parentId: null, timestamp: "", systemPrompt: "s", task: "t" };
			fs.writeFileSync(childFile, `${JSON.stringify(header)}\n${JSON.stringify(init)}\n`);
			AgentRegistry.global().register({
				id: options.id,
				displayName: options.id,
				kind: "sub",
				parentId: OWNER,
				session: null,
				sessionFile: childFile,
				status: "idle",
			});
			return resultFor(options);
		});
	}

	/** The first turn of `topicId`, once it ran. */
	async function firstTurnOf(topicId: string): Promise<ExecutorOptions> {
		await manager.getJob(`${topicId}-t1`)?.promise;
		const turn = firstTurns.find(options => options.id === topicId);
		if (!turn) throw new Error(`no first turn for ${topicId}`);
		return turn;
	}

	function user(text: string): string {
		return sm.appendMessage({ role: "user", content: text, timestamp: Date.now() });
	}

	function assistant(text: string): string {
		return sm.appendMessage(createAssistantMessage(text));
	}

	function customEntries<T>(type: "custom" | "custom_message", customType: string): T[] {
		return sm
			.getEntries()
			.filter(entry => entry.type === type && "customType" in entry && entry.customType === customType) as T[];
	}

	beforeEach(async () => {
		tempDir = TempDir.createSync("@omp-topic-suggestions-");
		firstTurns = [];
		hosts = [];
		queuedTurns = [];
		turnPending = false;
		manager = new AsyncJobManager({});
		sm = SessionManager.create(tempDir.path(), tempDir.path());
		inboxSm = sm;
		inboxEpoch = 0;
		stateWrite = undefined;
		user("hello");
		await sm.ensureOnDisk();
		mockFirstTurns();
	});

	afterEach(async () => {
		vi.restoreAllMocks();
		for (const host of hosts) host.dispose();
		await manager.dispose({ timeoutMs: 100 });
		VibeSessionRegistry.resetGlobalForTests();
		resetTopicsRegistryForTests();
		AgentRegistry.resetGlobalForTests();
		IrcBus.resetGlobalForTests();
		InternalUrlRouter.resetForTests();
		tempDir.removeSync();
	});

	it("records a suggestion as one visible entry naming Inbox messages, and creates nothing", async () => {
		const host = hostWith();
		const ask = user("Can you look at the Diokan term sheet?");
		const answer = assistant("Sure, the term sheet has three open points.");
		const suggestion = host.suggest({ title: " Diokan term sheet ", reason: "A separate deal.", messageIds: [ask, answer] });
		await Bun.sleep(0);

		const [entry] = customEntries<CustomMessageEntry>("custom_message", TOPIC_SUGGESTION_TYPE);
		expect(entry).toMatchObject({
			type: "custom_message",
			customType: "topic-suggestion",
			display: true,
			details: {
				id: suggestion.id,
				title: "Diokan term sheet",
				reason: "A separate deal.",
				messageIds: [ask, answer],
				createdMs: expect.any(Number),
			},
		});
		expect(typeof entry!.content).toBe("string");
		expect(entry!.content).toContain("Diokan term sheet");
		expect(suggestion.id).toMatch(/^sg_/);
		expect(Object.keys(entry!.details ?? {}).sort()).toEqual(["createdMs", "id", "messageIds", "reason", "title"]);
		expect(topicsRegistry().describe(parent())).toEqual([]);
		expect(firstTurns).toEqual([]);
	});

	it("refuses ids that are not user or assistant text of the current branch", async () => {
		const host = hostWith();
		const ask = user("Read the config file");
		const toolCall = sm.appendMessage({
			...createAssistantMessage(""),
			content: [{ type: "toolCall", id: "call-1", name: "read", arguments: { path: "/etc/config" } }],
		} as AssistantMessage);
		const toolOutput = sm.appendMessage({
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "read",
			content: [{ type: "text", text: "file body" }],
			isError: false,
			timestamp: Date.now(),
		});
		const elsewhere = assistant("an answer on a branch left behind");
		sm.branch(toolOutput);
		const current = assistant("The config sets three flags.");

		for (const bad of [toolCall, toolOutput, elsewhere, "no-such-entry"]) {
			expect(await codeOf(() => host.suggest({ title: "Config", reason: "r", messageIds: [ask, bad] }))).toBe(
				"invalid_message_ids",
			);
		}
		expect(await codeOf(() => host.suggest({ title: "Config", reason: "r", messageIds: [] }))).toBe(
			"invalid_message_ids",
		);
		expect(customEntries("custom_message", TOPIC_SUGGESTION_TYPE)).toEqual([]);
		expect(host.suggest({ title: "Config", reason: "r", messageIds: [ask, current] }).messageIds).toEqual([
			ask,
			current,
		]);
	});

	it("folds state entries: dismiss, dismiss again, accept after dismiss, accept again, dismiss refused", async () => {
		const host = hostWith();
		const ask = user("Plan the offsite");
		const { id } = host.suggest({ title: "Offsite", reason: "Its own project.", messageIds: [ask] });
		await Bun.sleep(0);

		expect(await host.dismissSuggestion({ suggestionId: id, by: "owner" })).toEqual({ alreadyDismissed: false });
		expect(await host.dismissSuggestion({ suggestionId: id, by: "owner" })).toEqual({ alreadyDismissed: true });
		expect(host.suggestions()).toEqual([]);

		const accepted = await host.acceptSuggestion({ suggestionId: id, by: "guest Anna" });
		expect(accepted).toEqual({ topicId: expect.any(String), alreadyAccepted: false });
		expect(await host.acceptSuggestion({ suggestionId: id, by: "owner" })).toEqual({
			topicId: accepted.topicId,
			alreadyAccepted: true,
		});
		expect(await codeOf(() => host.dismissSuggestion({ suggestionId: id, by: "owner" }))).toBe("already_accepted");

		const states = customEntries<CustomEntry>("custom", TOPIC_SUGGESTION_STATE_TYPE).map(entry => entry.data);
		expect(states).toEqual([
			{ id, state: "dismissed", by: "owner", atMs: expect.any(Number) },
			{ id, state: "accepted", topicId: accepted.topicId, by: "guest Anna", atMs: expect.any(Number) },
		]);
		expect(foldSuggestions(sm.getEntries()).get(id)).toMatchObject({ state: "accepted", topicId: accepted.topicId });
		expect(topicsRegistry().describe(parent()).map(topic => topic.id)).toEqual([accepted.topicId]);

		expect(await codeOf(() => host.acceptSuggestion({ suggestionId: "sg_unknown", by: "owner" }))).toBe(
			"suggestion_not_found",
		);
		expect(await codeOf(() => host.dismissSuggestion({ suggestionId: "sg_unknown", by: "owner" }))).toBe(
			"suggestion_not_found",
		);
	});

	it("seeds an accepted topic with exactly the suggestion's own messages", async () => {
		const host = hostWith();
		const first = user("first point about pricing");
		user("unrelated question");
		const second = assistant("second point about pricing");
		const { id } = host.suggest({ title: "Pricing", reason: "r", messageIds: [first, second] });

		const { topicId } = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		const task = (await firstTurnOf(topicId)).task;
		expect(task).toContain("first point about pricing");
		expect(task).toContain("second point about pricing");
		expect(task).not.toContain("unrelated question");
	});

	it("seeds the newest 30 messages verbatim, summarises the older ones once, and never copies tool calls or output", async () => {
		const summarize = vi.fn<SeedSummarizer>(async () => "SUMMARY OF EARLIER");
		const host = hostWith(summarize);
		const ids: string[] = [];
		for (let n = 1; n <= 35; n++) {
			ids.push(n % 2 ? user(`message ${n} from owner`) : assistant(`message ${n} from inbox`));
			if (n === 20) {
				ids.pop();
				ids.push(
					sm.appendMessage({
						...createAssistantMessage("message 20 from inbox"),
						content: [
							{ type: "text", text: "message 20 from inbox" },
							{ type: "toolCall", id: "call-9", name: "read", arguments: { path: "/SECRET-ARG" } },
						],
					} as AssistantMessage),
				);
				sm.appendMessage({
					role: "toolResult",
					toolCallId: "call-9",
					toolName: "read",
					content: [{ type: "text", text: "TOOL-OUTPUT-BODY" }],
					isError: false,
					timestamp: Date.now(),
				});
			}
		}
		const { id } = host.suggest({ title: "Long thread", reason: "REASON-TEXT", messageIds: ids });
		const { topicId } = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		const task = (await firstTurnOf(topicId)).task;

		expect(summarize).toHaveBeenCalledTimes(1);
		const summarised = summarize.mock.calls[0]![0];
		for (let n = 1; n <= 5; n++) expect(summarised).toContain(`message ${n} from`);
		expect(summarised).not.toContain("message 6 from");
		expect(task).toContain("Summary of 5 earlier messages:\nSUMMARY OF EARLIER");
		expect(task).toContain("REASON-TEXT");
		for (let n = 1; n <= 5; n++) expect(task).not.toContain(`message ${n} from`);
		for (let n = 6; n <= 35; n++) expect(task).toContain(`message ${n} from`);
		expect(task).toContain('<message role="assistant">\nmessage 20 from inbox\n</message>');
		expect(task).not.toContain("SECRET-ARG");
		expect(task).not.toContain("TOOL-OUTPUT-BODY");
	});

	it("leaves tool output out of a seed even when its id is named", async () => {
		const ask = user("what is in the file?");
		const output = sm.appendMessage({
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "read",
			content: [{ type: "text", text: "TOOL-OUTPUT-BODY" }],
			isError: false,
			timestamp: Date.now(),
		});
		const seed = await buildTopicSeed({
			entries: sm.getEntries(),
			messageIds: [ask, output],
			summarize: async () => "unused",
		});
		expect(seed).toContain("what is in the file?");
		expect(seed).not.toContain("TOOL-OUTPUT-BODY");
	});

	it("starts the topic without the older messages when their summary fails", async () => {
		for (const summarize of [
			async () => {
				throw new Error("model down");
			},
			async () => undefined,
		] satisfies SeedSummarizer[]) {
			const host = hostWith(summarize);
			const ids = Array.from({ length: 33 }, (_, n) => user(`numbered ${String(n + 1).padStart(2, "0")}`));
			const { id } = host.suggest({ title: `Fallback ${firstTurns.length}`, reason: "r", messageIds: ids });
			const { topicId } = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
			const task = (await firstTurnOf(topicId)).task;
			expect(task).toContain("3 earlier messages not summarised.");
			for (const n of ["01", "02", "03"]) expect(task).not.toContain(`numbered ${n}`);
			for (const n of ["04", "33"]) expect(task).toContain(`numbered ${n}`);
		}
	});

	it("returns from accept before the seed is ready, and starts the first turn only once it is", async () => {
		const summary = Promise.withResolvers<string | undefined>();
		const host = hostWith(() => summary.promise);
		const ids = Array.from({ length: 31 }, (_, n) => user(`line ${n}`));
		const { id } = host.suggest({ title: "Slow seed", reason: "r", messageIds: ids });

		const accepted = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		expect(accepted.alreadyAccepted).toBe(false);
		// The first turn's job is registered but waits on the seed; a macrotask turn lets anything unblocked run.
		await Bun.sleep(0);
		expect(firstTurns).toEqual([]);
		expect(topicsRegistry().describe(parent())).toMatchObject([{ id: accepted.topicId, state: "starting" }]);

		summary.resolve("LATE SUMMARY");
		expect((await firstTurnOf(accepted.topicId)).task).toContain("LATE SUMMARY");
	});

	it("notes in the Inbox where an accepted suggestion continues", async () => {
		const host = hostWith();
		const ask = user("Book the venue");
		const { id } = host.suggest({ title: "Venue", reason: "r", messageIds: [ask] });
		const { topicId } = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		await Bun.sleep(0);

		const notes = customEntries<CustomMessageEntry>("custom_message", TOPIC_CONTINUED_TYPE);
		expect(notes).toMatchObject([
			{
				display: true,
				content: "Continued in topic Venue",
				details: { topicId, title: "Venue", suggestionId: id, messageIds: [ask] },
			},
		]);
	});

	it("seeds topics_new from eligible Inbox messages, with a note that names no suggestion", async () => {
		const host = hostWith();
		const ask = user("Draft the newsletter");
		const output = sm.appendMessage({
			role: "toolResult",
			toolCallId: "c",
			toolName: "read",
			content: [{ type: "text", text: "x" }],
			isError: false,
			timestamp: Date.now(),
		});
		expect(
			await codeOf(() =>
				runRpcTopicsCommand(host, {
					type: "topics_new",
					title: "Newsletter",
					message: "go",
					seedMessageIds: [ask, output],
				}),
			),
		).toBe("invalid_message_ids");
		expect(firstTurns).toEqual([]);

		const { id: topicId } = await host.create({
			title: "Newsletter",
			message: "Write the first draft.",
			seedMessageIds: [ask],
		});
		const task = (await firstTurnOf(topicId)).task;
		expect(task).toMatch(/^<inbox-history>\n[\s\S]*Draft the newsletter[\s\S]*<\/inbox-history>\n\nWrite the first draft\.$/);
		await Bun.sleep(0);
		expect(customEntries<CustomMessageEntry>("custom_message", TOPIC_CONTINUED_TYPE)).toMatchObject([
			{ details: { topicId, title: "Newsletter", suggestionId: null, messageIds: [ask] } },
		]);

		const plain = await host.create({ title: "Plain", message: "just this" });
		expect((await firstTurnOf(plain.id)).task).toBe("just this");
		expect(customEntries("custom_message", TOPIC_CONTINUED_TYPE)).toHaveLength(1);
	});

	it("lists only open suggestions in topics_list", async () => {
		const host = hostWith();
		const ask = user("three subjects");
		const open = host.suggest({ title: "Open one", reason: "r1", messageIds: [ask] });
		const dismissed = host.suggest({ title: "Dismissed one", reason: "r2", messageIds: [ask] });
		const accepted = host.suggest({ title: "Accepted one", reason: "r3", messageIds: [ask] });
		await host.dismissSuggestion({ suggestionId: dismissed.id, by: "owner" });
		await host.acceptSuggestion({ suggestionId: accepted.id, by: "owner" });

		expect(await runRpcTopicsCommand(host, { type: "topics_list" })).toEqual({
			topics: expect.any(Array),
			suggestions: [{ id: open.id, title: "Open one", reason: "r1", messageIds: [ask], createdMs: open.createdMs }],
		});
	});

	it("puts open suggestions and the latest ten dismissed titles on the Inbox board only", async () => {
		const host = hostWith();
		const ask = user("many subjects");
		const open = host.suggest({ title: "Still open", reason: "waiting", messageIds: [ask] });
		const accepted = host.suggest({ title: "Taken up", reason: "r", messageIds: [ask] });
		await host.acceptSuggestion({ suggestionId: accepted.id, by: "owner" });
		for (let n = 1; n <= 12; n++) {
			const { id } = host.suggest({ title: `Dropped ${n}`, reason: "r", messageIds: [ask] });
			await host.dismissSuggestion({ suggestionId: id, by: "owner" });
		}

		const board = formatTopicBoard(host.list(), "inbox", host.suggestionBoard())!;
		expect(board).toContain(`- Still open (suggestion ${open.id}, 1 messages): waiting`);
		expect(board).toContain("Dismissed: do not suggest these again unless the user asks\n- Dropped 12\n");
		for (let n = 3; n <= 12; n++) expect(board).toContain(`- Dropped ${n}\n`);
		expect(board).not.toContain("Dropped 2\n");
		expect(board).not.toContain("Dropped 1\n");
		expect(board).not.toContain("(suggestion " + accepted.id);

		const topicBoard = formatTopicBoard(host.list(), "topic", host.suggestionBoard());
		expect(topicBoard).not.toContain("Still open");
		expect(topicBoard).not.toContain("Dismissed:");

		const onlySuggestions = formatTopicBoard([], "inbox", { open: [open], dismissed: [] });
		expect(onlySuggestions).toContain("Still open");
		expect(formatTopicBoard([], "inbox", { open: [], dismissed: [] })).toBeUndefined();
	});

	it("signals a change on every suggestion, accept and dismiss", async () => {
		const host = hostWith();
		const changes: TopicChange[] = [];
		host.onChange(change => changes.push(change));
		const ask = user("two subjects");
		const first = host.suggest({ title: "One", reason: "r", messageIds: [ask] });
		const second = host.suggest({ title: "Two", reason: "r", messageIds: [ask] });
		await Bun.sleep(0);
		await host.dismissSuggestion({ suggestionId: second.id, by: "owner" });
		await host.acceptSuggestion({ suggestionId: first.id, by: "owner" });

		const reasons = changes
			.filter(change => change.reason.startsWith("suggest"))
			.map(change => [change.id, change.reason]);
		expect(reasons).toEqual([
			[first.id, "suggested"],
			[second.id, "suggested"],
			[second.id, "suggestion-dismissed"],
			[first.id, "suggestion-accepted"],
		]);
		expect(changes.every(change => change.ownerId === OWNER)).toBe(true);
	});

	it("deduplicates the whole accept: a repeat shares its state write and its failure, a retry reuses the topic", async () => {
		const host = hostWith();
		const ask = user("Plan the offsite");
		const { id } = host.suggest({ title: "Offsite", reason: "r", messageIds: [ask] });
		const write = Promise.withResolvers<void>();
		stateWrite = () => write.promise;

		const first = host.acceptSuggestion({ suggestionId: id, by: "owner" });
		const repeat = host.acceptSuggestion({ suggestionId: id, by: "guest Anna" });
		write.reject(new Error("disk full"));
		const outcomes = await Promise.allSettled([first, repeat]);
		expect(outcomes.map(outcome => (outcome.status === "rejected" ? String(outcome.reason) : outcome.status))).toEqual([
			"Error: disk full",
			"Error: disk full",
		]);
		expect(customEntries("custom", TOPIC_SUGGESTION_STATE_TYPE)).toEqual([]);
		expect(customEntries("custom_message", TOPIC_CONTINUED_TYPE)).toEqual([]);

		stateWrite = undefined;
		const retried = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		expect(retried.alreadyAccepted).toBe(false);
		expect(topicsRegistry().describe(parent()).map(topic => topic.id)).toEqual([retried.topicId]);
		expect(customEntries<CustomEntry>("custom", TOPIC_SUGGESTION_STATE_TYPE).map(entry => entry.data)).toMatchObject([
			{ id, state: "accepted", topicId: retried.topicId },
		]);
	});

	it("refuses an accept whose Inbox session changed, writing nothing into the new one", async () => {
		const host = hostWith();
		const changes: TopicChange[] = [];
		host.onChange(change => changes.push(change));
		const ask = user("Book the venue");
		const { id } = host.suggest({ title: "Venue", reason: "r", messageIds: [ask] });
		const next = SessionManager.create(tempDir.path(), tempDir.path());
		const registry = topicsRegistry();
		const spawn = registry.spawn.bind(registry);
		// `/new` lands while the topic is being started.
		vi.spyOn(registry, "spawn").mockImplementation(async (...args) => {
			const spawned = await spawn(...args);
			switchInbox(next);
			return spawned;
		});

		expect(await codeOf(() => host.acceptSuggestion({ suggestionId: id, by: "owner" }))).toBe("suggestion_not_found");
		await Bun.sleep(0);
		for (const manager of [sm, next]) {
			const written = manager
				.getEntries()
				.filter(
					entry =>
						(entry.type === "custom" && entry.customType === TOPIC_SUGGESTION_STATE_TYPE) ||
						(entry.type === "custom_message" && entry.customType === TOPIC_CONTINUED_TYPE),
				);
			expect(written).toEqual([]);
		}
		expect(changes.some(change => change.reason === "suggestion-accepted")).toBe(false);
	});

	it("settles a topic whose seed fails, never leaving the rejection unhandled, even while the spawn write waits", async () => {
		const unhandled: unknown[] = [];
		const onUnhandled = (reason: unknown) => unhandled.push(reason);
		process.on("unhandledRejection", onUnhandled);
		try {
			const registry = topicsRegistry();
			// The parent write of the spawn waits; the seed fails meanwhile.
			const write = Promise.withResolvers<void>();
			const ensureOnDisk = sm.ensureOnDisk.bind(sm);
			vi.spyOn(sm, "ensureOnDisk").mockImplementation(async () => {
				await write.promise;
				await ensureOnDisk();
			});
			const seed = vi.fn(() => Promise.reject(new Error("seed failed")));
			const spawning = registry.spawn(parent(), { cli: "good", name: "Seedless", prompt: "go", title: "Seedless", seed });
			// Two macrotask turns: anything rejected and unowned by now is reported as unhandled.
			await macrotask();
			await macrotask();
			expect(seed).not.toHaveBeenCalled();
			write.resolve();
			const { id, jobId } = await spawning;
			await manager.getJob(jobId)?.promise.catch(() => {});
			await macrotask();
			const topic = registry.describe(parent()).find(entry => entry.id === id);
			expect(seed).toHaveBeenCalledTimes(1);
			expect(topic?.state).not.toBe("starting");
			expect(topic?.turnStartedAt).toBeUndefined();
			expect(firstTurns).toEqual([]);
			expect(unhandled).toEqual([]);
		} finally {
			process.off("unhandledRejection", onUnhandled);
		}
	});

	/** A fork of `sm`: a new session holding a copy of its suggestion entries. */
	function forkOfInbox(): SessionManager {
		const fork = SessionManager.create(tempDir.path(), tempDir.path());
		for (const entry of sm.getEntries()) {
			if (entry.type === "custom_message" && entry.customType === TOPIC_SUGGESTION_TYPE) {
				fork.appendCustomMessageEntry(entry.customType, entry.content, true, entry.details, "agent");
			}
		}
		return fork;
	}

	function acceptedStates(journal: SessionManager): unknown[] {
		return journal
			.getEntries()
			.filter(entry => entry.type === "custom" && entry.customType === TOPIC_SUGGESTION_STATE_TYPE);
	}

	it("refuses a retried accept in a fork after the accepted state failed to save, writing nothing there", async () => {
		const host = hostWith();
		const ask = user("Plan the offsite");
		const { id } = host.suggest({ title: "Offsite", reason: "r", messageIds: [ask] });
		stateWrite = () => Promise.reject(Object.assign(new Error("ENOSPC: no space left on device"), { code: "ENOSPC" }));
		await expect(host.acceptSuggestion({ suggestionId: id, by: "owner" })).rejects.toThrow("ENOSPC");
		expect(acceptedStates(sm)).toEqual([]);

		stateWrite = undefined;
		const fork = forkOfInbox();
		switchInbox(fork);
		expect(host.suggestions().map(open => open.id)).toEqual([id]);
		expect(await codeOf(() => host.acceptSuggestion({ suggestionId: id, by: "owner" }))).toBe("suggestion_not_found");
		expect(acceptedStates(fork)).toEqual([]);
		expect(acceptedStates(sm)).toEqual([]);
	});

	it("refuses a retried accept after /new and a reopen of the original Inbox: no false success", async () => {
		const host = hostWith();
		const ask = user("Book the venue");
		const { id } = host.suggest({ title: "Venue", reason: "r", messageIds: [ask] });
		const write = Promise.withResolvers<void>();
		stateWrite = () => write.promise;
		const accepting = host.acceptSuggestion({ suggestionId: id, by: "owner" });
		await Bun.sleep(0);
		// `/new` completes while the accepted state is being saved, then the original Inbox is reopened.
		switchInbox(SessionManager.create(tempDir.path(), tempDir.path()));
		write.resolve();
		expect(await codeOf(() => accepting)).toBe("suggestion_not_found");
		stateWrite = undefined;
		switchInbox(sm);

		expect(await codeOf(() => host.acceptSuggestion({ suggestionId: id, by: "owner" }))).toBe("suggestion_not_found");
		expect(acceptedStates(sm)).toEqual([]);
	});

	it("queues one Organise turn naming the Inbox messages, and refuses another while it is pending", async () => {
		const host = hostWith();
		const ask = user("lots going on");
		expect(await host.organise({ by: "owner" })).toEqual({ queued: false });
		expect(queuedTurns).toHaveLength(1);
		expect(queuedTurns[0]).toMatchObject({ customType: TOPIC_ORGANISE_TYPE, details: { by: "owner" } });
		expect(queuedTurns[0]!.content).toContain("call `topic_suggest` 3–5 times");
		expect(queuedTurns[0]!.content).toContain(`- ${ask} · user · "lots going on"`);

		expect(await codeOf(() => host.organise({ by: "owner" }))).toBe("organise_running");
		turnPending = false;
		expect(await host.organise({ by: "owner" })).toEqual({ queued: false });
	});
});

describe("seed summary model", () => {
	it("uses the smol role's default priority chain when no smol role is configured", async () => {
		registerMockApi();
		const mock = createMockModel({ id: "claude-haiku-4-5", handler: { content: ["CONDENSED"] } });
		const registry = {
			getAvailable: () => [mock.model],
			resolver: () => async () => "test-key",
		} as unknown as ModelRegistry;
		const settings = Settings.isolated();
		expect(settings.getModelRole("smol")).toBeUndefined();
		const session = { settings, modelRegistry: registry } as unknown as ToolSession;

		expect(await smolSeedSummarizer(session)("<message>older</message>")).toBe("CONDENSED");
		expect(mock.calls).toHaveLength(1);
	});
});
