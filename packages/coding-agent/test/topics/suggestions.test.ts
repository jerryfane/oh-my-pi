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
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
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
import { InboxEpochs, StaleTranscriptError, throwIfStale } from "@oh-my-pi/pi-coding-agent/session/inbox-epoch";
import { afterLatestReset, type InboxMessage, type TopicsInbox } from "@oh-my-pi/pi-coding-agent/topics/inbox";
import { buildTopicSeed, type SeedSummarizer, smolSeedSummarizer } from "@oh-my-pi/pi-coding-agent/topics/seed";
import {
	foldSuggestions,
	TOPIC_CONTINUED_TYPE,
	TOPIC_ORGANISE_TYPE,
	TOPIC_START_FAILED_TYPE,
	TOPIC_SUGGESTION_STATE_TYPE,
	TOPIC_SUGGESTION_TYPE,
	TopicsError,
} from "@oh-my-pi/pi-coding-agent/topics/suggestions";
import { resetTopicsRegistryForTests, topicsRegistry } from "@oh-my-pi/pi-coding-agent/topics/workers";
import { VibeSessionRegistry } from "@oh-my-pi/pi-coding-agent/vibe/runtime";
import type { SingleResult } from "@oh-my-pi/pi-coding-agent/task/task-details";
import { TempDir } from "@oh-my-pi/pi-utils";
import { VIBE_LIFECYCLE_CUSTOM_TYPE } from "@oh-my-pi/pi-coding-agent/vibe/lifecycle";
import { createAssistantMessage, createInMemoryAuthStorage } from "../helpers/agent-session-setup";

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
	/** The Inbox's epochs: switching sessions (a `/new`, a fork, a reopen) ends one and starts the next. */
	let inboxEpochs: InboxEpochs;
	/** Awaited by every state write before it saves; a rejection fails the write. */
	let stateWrite: (() => Promise<void>) | undefined;

	/** Open `next` in the Inbox: a `/new`, a fork, or a reopen of an earlier session. */
	function switchInbox(next: SessionManager): void {
		inboxEpochs.end();
		inboxSm = next;
		inboxEpochs.start();
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
	 * The Inbox over `inboxSm` (`sm` unless a test switches sessions), checking the binding where each effect
	 * happens as the session does: notes are saved at once (an idle Inbox); a state write awaits `stateWrite`
	 * first and, like the real one, adds nothing when it fails or another transcript is open by then.
	 */
	function inbox(): TopicsInbox {
		return {
			capture: () => inboxEpochs.capture(),
			branch: () => afterLatestReset(inboxSm.getBranch()),
			entries: () => inboxSm.getEntriesView(),
			appendState: async (epoch, customType, data) => {
				throwIfStale(epoch.signal);
				await stateWrite?.();
				throwIfStale(epoch.signal);
				inboxSm.appendCustomEntry(customType, data);
			},
			appendNote: async (epoch, { customType, content, details }) => {
				if (epoch.signal.aborted) return false;
				inboxSm.appendCustomMessageEntry(customType, content, true, details, "agent");
				return true;
			},
			queueTurn: async (epoch, customType, render) => {
				const rendered = render();
				throwIfStale(epoch.signal);
				queuedTurns.push({ customType, ...rendered });
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

	/** Run every topic's first turn the way the executor does (its request starts; leaves a child transcript, idle ref). */
	function mockFirstTurns(): void {
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async options => {
			firstTurns.push(options);
			// As the executor: a job aborted before it starts is refused before its first prompt.
			if (options.signal?.aborted) return { ...resultFor(options), exitCode: 1, aborted: true };
			options.onFirstTurn?.({ kind: "started" });
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

	/**
	 * The seed `topicId`'s first turn was launched with (`launchPreamble`), once it ran. It is handed to the executor
	 * apart from the task (the run's metadata), which never holds it; their order in the prompt actually sent is
	 * asserted on a real provider request (seeded-launch.test.ts).
	 */
	async function seedOf(topicId: string): Promise<string> {
		const { task, launchPreamble } = await firstTurnOf(topicId);
		expect(launchPreamble).toBeDefined();
		expect(task).not.toContain("<inbox-history>");
		return launchPreamble!.text;
	}

	/** The first turn of `topicId`, once it ran. */
	async function firstTurnOf(topicId: string): Promise<ExecutorOptions> {
		await manager.getJob(`${topicId}-t1`)?.promise;
		const turn = firstTurns.find(options => options.id === topicId);
		if (!turn) throw new Error(`no first turn for ${topicId}`);
		return turn;
	}

	/** Wait until `topicId`'s first turn ran and what follows its admission (the accepted state, the note) settled. */
	async function committed(topicId: string): Promise<void> {
		await firstTurnOf(topicId);
		await macrotask();
		await macrotask();
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
		inboxEpochs = new InboxEpochs(() => {});
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
		await committed(accepted.topicId);
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
		const seed = await seedOf(topicId);
		expect(seed).toContain("first point about pricing");
		expect(seed).toContain("second point about pricing");
		expect(seed).not.toContain("unrelated question");
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
		const seed = await seedOf(topicId);

		expect(summarize).toHaveBeenCalledTimes(1);
		const summarised = summarize.mock.calls[0]![0];
		for (let n = 1; n <= 5; n++) expect(summarised).toContain(`message ${n} from`);
		expect(summarised).not.toContain("message 6 from");
		expect(seed).toContain("Summary of 5 earlier messages:\nSUMMARY OF EARLIER");
		expect(seed).toContain("REASON-TEXT");
		for (let n = 1; n <= 5; n++) expect(seed).not.toContain(`message ${n} from`);
		for (let n = 6; n <= 35; n++) expect(seed).toContain(`message ${n} from`);
		expect(seed).toContain('<message role="assistant">\nmessage 20 from inbox\n</message>');
		expect(seed).not.toContain("SECRET-ARG");
		expect(seed).not.toContain("TOOL-OUTPUT-BODY");
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
		expect(seed.text).toContain("what is in the file?");
		expect(seed.text).not.toContain("TOOL-OUTPUT-BODY");
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
			const seed = await seedOf(topicId);
			expect(seed).toContain("3 earlier messages not summarised.");
			for (const n of ["01", "02", "03"]) expect(seed).not.toContain(`numbered ${n}`);
			for (const n of ["04", "33"]) expect(seed).toContain(`numbered ${n}`);
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
		expect(await seedOf(accepted.topicId)).toContain("LATE SUMMARY");
	});

	it("notes in the Inbox where an accepted suggestion continues", async () => {
		const host = hostWith();
		const ask = user("Book the venue");
		const { id } = host.suggest({ title: "Venue", reason: "r", messageIds: [ask] });
		const { topicId } = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		await committed(topicId);

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
		expect(await seedOf(topicId)).toMatch(/^<inbox-history>\n[\s\S]*Draft the newsletter[\s\S]*<\/inbox-history>$/);
		expect((await firstTurnOf(topicId)).task).toBe("Write the first draft.");
		await Bun.sleep(0);
		expect(customEntries<CustomMessageEntry>("custom_message", TOPIC_CONTINUED_TYPE)).toMatchObject([
			{ details: { topicId, title: "Newsletter", suggestionId: null, messageIds: [ask] } },
		]);

		const plain = await host.create({ title: "Plain", message: "just this" });
		expect(await firstTurnOf(plain.id)).toMatchObject({ task: "just this", launchPreamble: undefined });
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
		await committed((await host.acceptSuggestion({ suggestionId: first.id, by: "owner" })).topicId);

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

	it("joins a repeat accept while it is pending; a failed state write leaves the suggestion open, ends the topic, and a retry starts a fresh one", async () => {
		const host = hostWith();
		const ask = user("Plan the offsite");
		const { id } = host.suggest({ title: "Offsite", reason: "r", messageIds: [ask] });
		const write = Promise.withResolvers<void>();
		stateWrite = () => write.promise;

		const first = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		expect(await host.acceptSuggestion({ suggestionId: id, by: "guest Anna" })).toEqual({
			topicId: first.topicId,
			alreadyAccepted: true,
		});
		expect(host.suggestions()).toEqual([]);
		await firstTurnOf(first.topicId);
		write.reject(new Error("disk full"));
		await macrotask();
		await macrotask();
		expect(customEntries("custom", TOPIC_SUGGESTION_STATE_TYPE)).toEqual([]);
		expect(customEntries("custom_message", TOPIC_CONTINUED_TYPE)).toEqual([]);
		expect(host.suggestions().map(open => open.id)).toEqual([id]);
		expect(topicsRegistry().worker(parent(), first.topicId)).toBeUndefined();

		stateWrite = undefined;
		const retried = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		expect(retried.alreadyAccepted).toBe(false);
		expect(retried.topicId).not.toBe(first.topicId);
		await committed(retried.topicId);
		expect(customEntries<CustomEntry>("custom", TOPIC_SUGGESTION_STATE_TYPE).map(entry => entry.data)).toMatchObject([
			{ id, state: "accepted", topicId: retried.topicId },
		]);
		expect(topicsRegistry().worker(parent(), retried.topicId)).toBeDefined();
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

	it("never writes a stale accept into the Inbox that replaced its own while the seed's room was checked (#21 review)", async () => {
		const authStorage = createInMemoryAuthStorage();
		authStorage.keys.setRuntime("mock", "test-key");
		const modelRegistry = new ModelRegistry(authStorage);
		vi.spyOn(modelRegistry, "getAvailable").mockReturnValue([createMockModel({})]);
		const lookup = Promise.withResolvers<void>();
		const looking = Promise.withResolvers<void>();
		const getApiKey = modelRegistry.getApiKey.bind(modelRegistry);
		vi.spyOn(modelRegistry, "getApiKey").mockImplementation(async (...args) => {
			looking.resolve();
			await lookup.promise;
			return getApiKey(...args);
		});
		// The Inbox's tool session follows the open transcript, as the real one does across `/new`.
		const live = {
			...parent(),
			settings: Settings.isolated({ modelRoles: { default: "mock/mock-model", task: "mock/mock-model" } }),
			modelRegistry,
			getActiveModelString: () => "mock/mock-model",
			getSessionId: () => inboxSm.getSessionId(),
			getSessionFile: () => inboxSm.getSessionFile() ?? null,
			get sessionManager() {
				return inboxSm;
			},
		} as unknown as ToolSession;
		const host = createTopicsHost(live, { inbox: inbox(), summarize: async () => "SUMMARY" });
		hosts.push(host);
		const changes: TopicChange[] = [];
		host.onChange(change => changes.push(change));
		const { id } = host.suggest({ title: "Stale venue", reason: "r", messageIds: [user("Book the venue")] });
		// Accept waits in the seed's room check (the start model's credential lookup) while `/new` lands.
		const accepting = codeOf(() => host.acceptSuggestion({ suggestionId: id, by: "owner" }));
		await looking.promise;
		const next = SessionManager.create(tempDir.path(), tempDir.path());
		await next.ensureOnDisk();
		switchInbox(next);
		lookup.resolve();

		expect(await accepting).toBe("suggestion_not_found");
		await macrotask();
		await macrotask();
		expect(firstTurns).toEqual([]);
		expect(changes.filter(change => change.reason === "spawned")).toEqual([]);
		for (const journal of [sm, next]) {
			const lifecycle = journal
				.getEntries()
				.filter(entry => entry.type === "custom" && entry.customType === VIBE_LIFECYCLE_CUSTOM_TYPE);
			expect(lifecycle).toEqual([]);
		}
		const nextFile = next.getSessionFile();
		expect(nextFile && fs.existsSync(nextFile) ? fs.readFileSync(nextFile, "utf8") : "").not.toContain("Stale venue");
		expect(topicsRegistry().describe(live)).toEqual([]);
	});

	it("watches the first turn of seeded launches only (#21 review)", async () => {
		const host = hostWith();
		const plain = await host.create({ title: "Plain", message: "go" });
		const seeded = await host.create({ title: "Seeded", message: "go", seedMessageIds: [user("Plan the offsite")] });
		for (const { id } of [plain, seeded]) await manager.getJob(`${id}-t1`)?.promise;

		const watched = (id: string) => firstTurns.find(turn => turn.id === id)?.onFirstTurn !== undefined;
		expect({ plain: watched(plain.id), seeded: watched(seeded.id) }).toEqual({ plain: false, seeded: true });
	});

	it("spawns nothing for a seed whose transcript was already replaced: no worker, journal entry or first turn (#21 review)", async () => {
		const replaced = new AbortController();
		replaced.abort();
		const firstTurn = vi.fn();
		const spawning = topicsRegistry().spawn(parent(), {
			cli: "good",
			name: "Stale",
			prompt: "go",
			title: "Stale title",
			seed: {
				signal: replaced.signal,
				build: async () => ({ text: "", fit: () => ({ text: "", tokens: 0 }) }),
				firstTurn,
			},
		});

		await expect(spawning).rejects.toBeInstanceOf(StaleTranscriptError);
		expect(topicsRegistry().describe(parent())).toEqual([]);
		expect(customEntries("custom", VIBE_LIFECYCLE_CUSTOM_TYPE)).toEqual([]);
		expect(firstTurns).toEqual([]);
		expect(firstTurn).not.toHaveBeenCalled();
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
			const firstTurn = vi.fn();
			const spawning = registry.spawn(parent(), {
				cli: "good",
				name: "Seedless",
				prompt: "go",
				title: "Seedless",
				seed: { signal: new AbortController().signal, build: seed, firstTurn },
			});
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
			expect(firstTurn.mock.calls).toEqual([[{ kind: "start-failed", code: "start_failed", error: "seed failed" }]]);
			expect(unhandled).toEqual([]);
		} finally {
			process.off("unhandledRejection", onUnhandled);
		}
	});

	/** A fork of `sm`: a new session with the same entries (ids kept), as `/fork` makes. */
	async function forkOfInbox(): Promise<SessionManager> {
		const fork = await SessionManager.open(sm.getSessionFile()!, tempDir.path());
		fork.createBranchedSession(fork.getLeafId()!);
		return fork;
	}

	function acceptedStates(journal: SessionManager): unknown[] {
		return journal
			.getEntries()
			.filter(entry => entry.type === "custom" && entry.customType === TOPIC_SUGGESTION_STATE_TYPE);
	}

	it("never links a topic its Inbox does not list: an ENOSPC accepted-state write, then a fork, then a retry there", async () => {
		const host = hostWith();
		const ask = user("Plan the offsite");
		const { id } = host.suggest({ title: "Offsite", reason: "r", messageIds: [ask] });
		stateWrite = () => Promise.reject(Object.assign(new Error("ENOSPC: no space left on device"), { code: "ENOSPC" }));
		const first = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		await committed(first.topicId);
		expect(acceptedStates(sm)).toEqual([]);
		expect(topicsRegistry().worker(parent(), first.topicId)).toBeUndefined();

		stateWrite = undefined;
		await sm.flush();
		const fork = await forkOfInbox();
		switchInbox(fork);
		expect(host.suggestions().map(open => open.id)).toEqual([id]);
		const retried = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		expect(retried.topicId).not.toBe(first.topicId);
		await committed(retried.topicId);
		expect(acceptedStates(fork)).toMatchObject([{ data: { id, state: "accepted", topicId: retried.topicId } }]);
		expect(topicsRegistry().worker(parent(), retried.topicId)).toBeDefined();
		expect(acceptedStates(sm)).toEqual([]);
	});

	it("never links an accept whose commit crossed /new; after a reopen, a retry starts a fresh live topic", async () => {
		const host = hostWith();
		const ask = user("Book the venue");
		const { id } = host.suggest({ title: "Venue", reason: "r", messageIds: [ask] });
		const write = Promise.withResolvers<void>();
		stateWrite = () => write.promise;
		const first = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		await firstTurnOf(first.topicId);
		// `/new` completes while the accepted state is being saved, then the original Inbox is reopened.
		switchInbox(SessionManager.create(tempDir.path(), tempDir.path()));
		write.resolve();
		await macrotask();
		await macrotask();
		stateWrite = undefined;
		switchInbox(sm);
		expect(acceptedStates(sm)).toEqual([]);
		expect(host.suggestions().map(open => open.id)).toEqual([id]);

		const retried = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		expect(retried).toEqual({ topicId: expect.any(String), alreadyAccepted: false });
		expect(retried.topicId).not.toBe(first.topicId);
		await committed(retried.topicId);
		expect(acceptedStates(sm)).toMatchObject([{ data: { topicId: retried.topicId } }]);
		expect(topicsRegistry().worker(parent(), retried.topicId)).toBeDefined();
	});

	it("leaves the suggestion open and links nothing when the first turn fails before dispatch, outside the seed (#12, #20 review)", async () => {
		// Checking the seed's room and the spawn each resolve the worker's model; building its first turn's options
		// then fails, after the seed, before dispatch: the job records the failure and fulfils its promise.
		let modelReads = 0;
		const failing = {
			...parent(),
			getActiveModelString: () => {
				if (modelReads++ > 1) throw new Error("no active model");
				return undefined;
			},
		} as unknown as ToolSession;
		const host = createTopicsHost(failing, { inbox: inbox(), summarize: async () => "SUMMARY" });
		hosts.push(host);
		const changes: TopicChange[] = [];
		host.onChange(change => changes.push(change));
		const ask = user("Plan the offsite");
		const { id } = host.suggest({ title: "Offsite", reason: "r", messageIds: [ask] });
		const { topicId } = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		await manager.getJob(`${topicId}-t1`)?.promise;
		await macrotask();
		await macrotask();

		expect(firstTurns).toEqual([]);
		expect(customEntries("custom", TOPIC_SUGGESTION_STATE_TYPE)).toEqual([]);
		expect(customEntries("custom_message", TOPIC_CONTINUED_TYPE)).toEqual([]);
		expect(host.suggestions().map(open => open.id)).toEqual([id]);
		expect(topicsRegistry().worker(failing, topicId)).toBeUndefined();
		// A start failure that is not the seed's size: the generic code, the error, one update, one note.
		expect(changes.filter(change => change.reason === "start-failed")).toMatchObject([
			{ id: topicId, suggestionId: id, code: "start_failed", noteWritten: true },
		]);
		expect(customEntries("custom_message", TOPIC_START_FAILED_TYPE)).toMatchObject([
			{ details: { topicId, suggestionId: id, code: "start_failed", error: "no active model" } },
		]);
	});

	it("sends the start-failed update even when its Inbox note cannot be written, saying so (#20 review)", async () => {
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async options => {
			firstTurns.push(options);
			options.onFirstTurn?.({ kind: "start-failed", code: "start_failed", error: "provider unavailable" });
			return { ...resultFor(options), exitCode: 1, error: "provider unavailable" };
		});
		const base = inbox();
		const failingNotes: TopicsInbox = {
			...base,
			appendNote: async (epoch, note) => {
				if (note.customType === TOPIC_START_FAILED_TYPE) throw new Error("EIO: i/o error");
				return base.appendNote(epoch, note);
			},
		};
		const host = createTopicsHost(parent(), { inbox: failingNotes, summarize: async () => "SUMMARY" });
		hosts.push(host);
		const changes: TopicChange[] = [];
		host.onChange(change => changes.push(change));
		const { id } = host.suggest({ title: "Offsite", reason: "r", messageIds: [user("Plan the offsite")] });
		const { topicId } = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		await manager.getJob(`${topicId}-t1`)?.promise;
		await macrotask();
		await macrotask();

		expect(changes.filter(change => change.reason === "start-failed")).toEqual([
			{
				id: topicId,
				ownerId: OWNER,
				parentSessionId: sm.getSessionId(),
				reason: "start-failed",
				suggestionId: id,
				code: "start_failed",
				noteWritten: false,
			},
		]);
		expect(customEntries("custom_message", TOPIC_START_FAILED_TYPE)).toEqual([]);
		expect(customEntries("custom", TOPIC_SUGGESTION_STATE_TYPE)).toEqual([]);
		expect(host.suggestions().map(open => open.id)).toEqual([id]);
		const retried = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		expect(retried.alreadyAccepted).toBe(false);
		expect(retried.topicId).not.toBe(topicId);
	});

	it("never links a worker cancelled before its first prompt was admitted; the suggestion stays open (#12 review)", async () => {
		const summary = Promise.withResolvers<string | undefined>();
		const host = hostWith(() => summary.promise);
		const ids = Array.from({ length: 33 }, (_, n) => user(`numbered ${n + 1}`));
		const { id } = host.suggest({ title: "Cancelled", reason: "r", messageIds: ids });
		const { topicId } = await host.acceptSuggestion({ suggestionId: id, by: "owner" });
		// The worker is cancelled while its seed is built; the executor then refuses the aborted job.
		const killing = topicsRegistry().kill(parent(), topicId);
		summary.resolve("SUMMARY");
		await killing;
		await manager.getJob(`${topicId}-t1`)?.promise;
		await macrotask();
		await macrotask();

		expect(firstTurns.map(turn => turn.signal?.aborted)).toEqual([true]);
		expect(customEntries("custom", TOPIC_SUGGESTION_STATE_TYPE)).toEqual([]);
		expect(customEntries("custom_message", TOPIC_CONTINUED_TYPE)).toEqual([]);
		expect(host.suggestions().map(open => open.id)).toEqual([id]);
	});

	it("clears only its own in-flight dismissal: an older epoch's ending never drops a newer one (#12 review)", async () => {
		const host = hostWith();
		const ask = user("Plan the offsite");
		const { id } = host.suggest({ title: "Offsite", reason: "r", messageIds: [ask] });
		const writes = [Promise.withResolvers<void>(), Promise.withResolvers<void>(), Promise.withResolvers<void>()];
		let started = 0;
		stateWrite = () => writes[started++]!.promise;

		const older = host.dismissSuggestion({ suggestionId: id, by: "owner" }).catch((error: Error) => error);
		// The same Inbox reopened: a new epoch, where the same suggestion is dismissed again.
		switchInbox(sm);
		const newer = host.dismissSuggestion({ suggestionId: id, by: "owner" });
		writes[0]!.resolve();
		expect(await older).toBeInstanceOf(Error);
		const repeat = host.dismissSuggestion({ suggestionId: id, by: "guest Anna" });
		for (const write of writes) write.resolve();

		expect(await newer).toEqual({ alreadyDismissed: false });
		expect(await repeat).toEqual({ alreadyDismissed: true });
		expect(started).toBe(2);
		expect(customEntries("custom", TOPIC_SUGGESTION_STATE_TYPE)).toHaveLength(1);
	});

	it("offers only suggestions whose messages are user or assistant text, as create and suggest require (#12 review)", async () => {
		const host = hostWith();
		const ask = user("what is in the file?");
		const output = sm.appendMessage({
			role: "toolResult",
			toolCallId: "call-1",
			toolName: "read",
			content: [{ type: "text", text: "TOOL-OUTPUT-BODY" }],
			isError: false,
			timestamp: Date.now(),
		});
		// Saved by an older build, or edited: it names a tool output.
		sm.appendCustomMessageEntry(TOPIC_SUGGESTION_TYPE, "Suggested topic: File", true, {
			id: "sg-tool",
			title: "File",
			reason: "r",
			messageIds: [ask, output],
			createdMs: 1,
		});
		expect(await codeOf(() => host.create({ title: "File", message: "go", seedMessageIds: [ask, output] }))).toBe(
			"invalid_message_ids",
		);

		expect(host.suggestions()).toEqual([]);
		expect(host.suggestionBoard().open).toEqual([]);
		expect(await codeOf(() => host.acceptSuggestion({ suggestionId: "sg-tool", by: "owner" }))).toBe(
			"suggestion_not_found",
		);
		expect(topicsRegistry().describe(parent())).toEqual([]);
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

		const origin = { sessionId: "s1", signal: new AbortController().signal };
		expect(await smolSeedSummarizer(session)("<message>older</message>", origin)).toBe("CONDENSED");
		expect(mock.calls).toHaveLength(1);
	});
});
