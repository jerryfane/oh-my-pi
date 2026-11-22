/**
 * Contract: suggestions and Organise on a real Inbox session (AgentSession,
 * mock model).
 *
 * - `topic_suggest` exists only in the Inbox (top-level session) with topics
 *   enabled: not with topics off, not in a subagent.
 * - The tool saves its suggestion once the Inbox turn ends (never inside it),
 *   refuses unknown ids with `invalid_message_ids`, and the next turn's board
 *   lists the open suggestion.
 * - Organise starts one hidden Inbox turn when idle; while busy it queues
 *   behind the running turn without interrupting it; a second Organise while
 *   one is queued or running is refused with `organise_running`.
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { Agent } from "@oh-my-pi/pi-agent-core";
import type { Context, Message } from "@oh-my-pi/pi-ai";
import { createMockModel, type MockResponse } from "@oh-my-pi/pi-ai/providers/mock";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { IrcBus } from "@oh-my-pi/pi-coding-agent/irc/bus";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { convertToLlm, USER_INTERRUPT_LABEL } from "@oh-my-pi/pi-coding-agent/session/messages";
import type { CustomMessageEntry, SessionEntry } from "@oh-my-pi/pi-coding-agent/session/session-entries";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { ExecutorOptions } from "@oh-my-pi/pi-coding-agent/task/executor";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { StaleTranscriptError } from "@oh-my-pi/pi-coding-agent/session/inbox-epoch";
import { createTopicsHost, type TopicsHost } from "@oh-my-pi/pi-coding-agent/topics/host";
import { sessionInbox } from "@oh-my-pi/pi-coding-agent/topics/inbox";
import type { SeedSummarizer } from "@oh-my-pi/pi-coding-agent/topics/seed";
import { TopicSuggestTool } from "@oh-my-pi/pi-coding-agent/topics/suggest-tool";
import { TOPIC_ORGANISE_TYPE, TOPIC_SUGGESTION_TYPE, TopicsError } from "@oh-my-pi/pi-coding-agent/topics/suggestions";
import { resetTopicsRegistryForTests } from "@oh-my-pi/pi-coding-agent/topics/workers";
import { TempDir } from "@oh-my-pi/pi-utils";
import type { IrcMessage } from "@oh-my-pi/pi-tui/tools/irc";
import { createAssistantMessage, createInMemoryAuthStorage } from "../helpers/agent-session-setup";

const OWNER = "Main";
const ORGANISE_MARK = "call `topic_suggest` 3–5 times";

const sessions: AgentSession[] = [];
const managers: AsyncJobManager[] = [];
const temps: TempDir[] = [];

afterEach(async () => {
	vi.restoreAllMocks();
	for (const session of sessions.splice(0)) await session.dispose();
	for (const manager of managers.splice(0)) await manager.dispose({ timeoutMs: 100 });
	resetTopicsRegistryForTests();
	AgentRegistry.resetGlobalForTests();
	IrcBus.resetGlobalForTests();
	for (const temp of temps.splice(0)) temp.removeSync();
});

function textOf(message: Message): string {
	const content = message.content;
	if (typeof content === "string") return content;
	return content.map(block => ("text" in block ? block.text : "")).join("");
}

function lastText(context: Context): string {
	const last = context.messages.at(-1);
	return last ? textOf(last) : "";
}

interface Inbox {
	session: AgentSession;
	manager: SessionManager;
	/** Messages of every model request. */
	requests: Message[][];
	host: TopicsHost;
	/** The Inbox's background jobs (topic turns). */
	jobs: AsyncJobManager;
}

/**
 * An Inbox session with topics on, its model answering through `respond`; `persist` = a session file on disk
 * (and topics persisted next to it); `summarize` condenses older seed messages.
 */
async function openInbox(
	respond: (context: Context) => MockResponse | Promise<MockResponse>,
	seed: (manager: SessionManager) => void = () => {},
	persist = false,
	summarize: SeedSummarizer = async () => "summary",
): Promise<Inbox> {
	const bundled = getBundledModel("anthropic", "claude-sonnet-4-5");
	if (!bundled) throw new Error("Expected bundled model");
	const temp = TempDir.createSync("@omp-inbox-suggest-");
	temps.push(temp);
	const manager = persist ? SessionManager.create(temp.path(), temp.path()) : SessionManager.inMemory(temp.path());
	seed(manager);
	const requests: Message[][] = [];
	const mock = createMockModel({
		provider: "anthropic",
		handler: context => {
			requests.push(structuredClone(context.messages));
			return respond(context);
		},
	});
	const jobs = new AsyncJobManager({});
	managers.push(jobs);
	const toolSession = {
		cwd: temp.path(),
		settings: Settings.isolated(),
		asyncJobManager: jobs,
		sessionManager: manager,
		getAgentId: () => OWNER,
		getSessionId: () => manager.getSessionId(),
		getSessionFile: () => (persist ? (manager.getSessionFile() ?? null) : null),
		getArtifactsDir: () => null,
		taskDepth: 0,
		enableLsp: false,
	} as unknown as ToolSession;
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: {
			model: bundled,
			systemPrompt: ["Inbox"],
			tools: [new TopicSuggestTool(() => session.getTopicsHost())],
			messages: manager.buildSessionContext().messages,
		},
		convertToLlm,
		streamFn: (model, context, options) => mock.stream(model, context, options),
	});
	const auth = createInMemoryAuthStorage();
	auth.keys.setRuntime("anthropic", "test-key");
	const session: AgentSession = new AgentSession({
		agent,
		sessionManager: manager,
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: new ModelRegistry(auth),
		createTopicsHost: () =>
			createTopicsHost(toolSession, { inbox: sessionInbox(session), summarize }),
	});
	sessions.push(session);
	const host = await session.getTopicsHost();
	if (!host) throw new Error("Expected a topics host");
	return { session, manager, requests, host, jobs };
}

async function codeOf(run: () => unknown): Promise<string | undefined> {
	try {
		await run();
	} catch (error) {
		if (error instanceof TopicsError) return error.code;
		throw error;
	}
	throw new Error("expected a refusal");
}

describe("topic_suggest registration", () => {
	async function toolNames(settings: Record<string, unknown>, subagent: boolean): Promise<string[]> {
		const temp = TempDir.createSync("@omp-topic-suggest-tools-");
		temps.push(temp);
		const authStorage = createInMemoryAuthStorage();
		const { session } = await createAgentSession({
			cwd: temp.path(),
			agentDir: temp.path(),
			authStorage,
			modelRegistry: new ModelRegistry(authStorage, path.join(temp.path(), "models.yml")),
			settings: Settings.isolated(settings),
			sessionManager: SessionManager.inMemory(),
			disableExtensionDiscovery: true,
			skills: [],
			contextFiles: [],
			promptTemplates: [],
			slashCommands: [],
			enableMCP: false,
			enableLsp: false,
			skipPythonPreflight: true,
			rules: [],
			preloadedCustomToolPaths: [],
			...(subagent ? { taskDepth: 1, parentTaskPrefix: "Helper", agentId: "Helper", parentAgentId: OWNER } : {}),
		});
		sessions.push(session);
		return session.getAllToolNames();
	}

	it("registers topic_suggest only in the Inbox with topics enabled", async () => {
		expect(await toolNames({ "topics.enabled": true }, false)).toContain("topic_suggest");
		expect(await toolNames({ "topics.enabled": false }, false)).not.toContain("topic_suggest");
		expect(await toolNames({ "topics.enabled": true }, true)).not.toContain("topic_suggest");
	}, 60_000);
});

describe("Inbox notes", () => {
	function notesOf(manager: SessionManager, customType: string): CustomMessageEntry[] {
		return manager
			.getEntries()
			.filter(
				(entry): entry is CustomMessageEntry => entry.type === "custom_message" && entry.customType === customType,
			);
	}

	it("saves two notes appended one after the other, each awaited", async () => {
		const inbox = await openInbox(() => ({ content: ["ok"] }));
		const box = sessionInbox(inbox.session);
		expect(await box.appendNote(box.capture(), { customType: "note-a", content: "first", details: {} })).toBe(true);
		expect(await box.appendNote(box.capture(), { customType: "note-b", content: "second", details: {} })).toBe(true);
		expect(notesOf(inbox.manager, "note-a")).toHaveLength(1);
		expect(notesOf(inbox.manager, "note-b")).toHaveLength(1);
	});

	/** Index of `predicate`'s first entry; -1 when none. */
	function indexOf(manager: SessionManager, predicate: (entry: SessionEntry) => boolean): number {
		return manager.getEntries().findIndex(predicate);
	}
	const isQuestion = (entry: SessionEntry): boolean =>
		entry.type === "message" && entry.message.role === "user" && textOf(entry.message) === "a question";
	const isNote = (entry: SessionEntry): boolean => entry.type === "custom_message" && entry.customType === "note-race";

	it("records a note appended while a turn is starting at that prompt's boundary, never as a steer", async () => {
		const inbox = await openInbox(() => ({ content: ["the answer"] }));
		const box = sessionInbox(inbox.session);
		const turn = inbox.session.prompt("a question");
		const saved = box.appendNote(box.capture(), { customType: "note-race", content: "NOTE-TEXT", details: {} });
		expect(await saved).toBe(true);
		await turn;
		await inbox.session.waitForIdle();

		// One request, the note in it as history ahead of the question: a steer lands after the question.
		expect(inbox.requests).toHaveLength(1);
		const sent = inbox.requests[0]!.map(textOf).join("\n");
		expect(sent.indexOf("NOTE-TEXT")).toBeGreaterThan(-1);
		expect(sent.indexOf("NOTE-TEXT")).toBeLessThan(sent.indexOf("a question"));
		expect(indexOf(inbox.manager, isNote)).toBeGreaterThan(-1);
		expect(indexOf(inbox.manager, isNote)).toBeLessThan(indexOf(inbox.manager, isQuestion));
	});

	it("records a note queued behind a wake that fails before streaming at that failure", async () => {
		const inbox = await openInbox(() => ({ content: ["unused"] }));
		const box = sessionInbox(inbox.session);
		const setup = Promise.withResolvers<void>();
		const failWake = Promise.withResolvers<never>();
		const prompt = vi.spyOn(inbox.session.agent, "prompt").mockImplementationOnce(() => {
			setup.resolve();
			return failWake.promise;
		});
		await inbox.session.deliverIrcMessage({ id: "m1", from: "Helper", to: OWNER, body: "ping", ts: Date.now() });
		await setup.promise;
		expect(inbox.session.isStreaming).toBe(true);
		const saved = box.appendNote(box.capture(), { customType: "note-race", content: "NOTE-TEXT", details: {} });
		failWake.reject(new Error("wake setup failed"));

		expect(await saved).toBe(true);
		expect(notesOf(inbox.manager, "note-race")).toMatchObject([{ display: true, content: "NOTE-TEXT" }]);
		prompt.mockRestore();
	});

	it("rejects a queued note when the session is disposed before its run settles", async () => {
		const inbox = await openInbox(() => ({ content: ["unused"] }));
		const box = sessionInbox(inbox.session);
		const setup = Promise.withResolvers<void>();
		// A wake whose setup never settles, not even on abort: the dispose drain gives up on it.
		vi.spyOn(inbox.session.agent, "prompt").mockImplementationOnce(() => {
			setup.resolve();
			return new Promise<never>(() => {});
		});
		await inbox.session.deliverIrcMessage({ id: "m1", from: "Helper", to: OWNER, body: "ping", ts: Date.now() });
		await setup.promise;
		const saved = box.appendNote(box.capture(), { customType: "note-race", content: "NOTE-TEXT", details: {} });
		await inbox.session.dispose({ drainTimeoutMs: 10 });
		sessions.splice(sessions.indexOf(inbox.session), 1);

		expect(await saved).toBe(false);
		expect(notesOf(inbox.manager, "note-race")).toEqual([]);
	});

	it("publishes a note recorded while idle as message_start and message_end", async () => {
		const inbox = await openInbox(() => ({ content: ["ok"] }));
		const events: string[] = [];
		inbox.session.subscribe(event => {
			if ((event.type === "message_start" || event.type === "message_end") && event.message.role === "custom") {
				events.push(`${event.type}:${event.message.customType}`);
			}
		});
		const box = sessionInbox(inbox.session);
		expect(await box.appendNote(box.capture(), { customType: "note-live", content: "x", details: {} })).toBe(true);
		expect(events).toEqual(["message_start:note-live", "message_end:note-live"]);
	});

	it("fails the note and never announces the suggestion when the session file cannot be written", async () => {
		let ask = "";
		const inbox = await openInbox(
			() => ({ content: ["ok"] }),
			manager => {
				ask = manager.appendMessage({ role: "user", content: "Let's plan the offsite", timestamp: 1 });
			},
		);
		// What a latched disk failure does: every later flush rethrows it.
		vi.spyOn(inbox.manager, "flush").mockRejectedValue(new Error("disk full"));
		const record = vi.spyOn(inbox.session, "recordWithoutTurn");
		const changes: string[] = [];
		inbox.host.onChange(change => changes.push(change.reason));

		const box = sessionInbox(inbox.session);
		expect(await box.appendNote(box.capture(), { customType: "note-fail", content: "x", details: {} })).toBe(false);
		const suggestion = inbox.host.suggest({ title: "Offsite", reason: "Its own project.", messageIds: [ask] });
		await expect(record.mock.results[1]!.value).rejects.toThrow("disk full");
		// The host settles the suggestion in the continuation of that rejected write.
		await Bun.sleep(0);
		expect(changes).not.toContain("suggested");
		// Visibility follows the session journal alone: listed exactly while its entry is there (write recovery may save it).
		const inJournal = inbox.manager
			.getEntries()
			.some(entry => entry.type === "custom_message" && entry.customType === TOPIC_SUGGESTION_TYPE);
		expect(inbox.host.suggestions().some(open => open.id === suggestion.id)).toBe(inJournal);
	});
});

describe("Organise turn tracked by its own message (#13 review)", () => {
	it("refuses a second Organise while the first is parked by an IRC wake: organise_running", async () => {
		const arrived = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const woke = Promise.withResolvers<void>();
		const wakeRelease = Promise.withResolvers<void>();
		const inbox = await openInbox(async context => {
			const text = lastText(context);
			if (text === "long question") {
				arrived.resolve();
				await release.promise;
				return { content: ["the full answer"] };
			}
			if (text.includes("ping")) {
				woke.resolve();
				await wakeRelease.promise;
				return { content: ["pong"] };
			}
			return { content: [text.includes(ORGANISE_MARK) ? "organised" : "ok"] };
		});
		const running = inbox.session.prompt("long question");
		await arrived.promise;
		expect(await inbox.host.organise({ by: "owner" })).toEqual({ queued: true });
		await inbox.session.abort({ reason: USER_INTERRUPT_LABEL });
		release.resolve();
		await running.catch(() => {});
		await inbox.session.waitForIdle();
		await inbox.session.deliverIrcMessage({
			id: "m1",
			from: "peer",
			to: "me",
			body: "ping",
			ts: Date.now(),
		} as IrcMessage);
		await woke.promise;
		// The first Organise is parked outside the agent queue, not removed.
		expect(inbox.session.agent.peekFollowUpQueue()).toHaveLength(0);
		const second = await codeOf(() => inbox.host.organise({ by: "owner" }));
		wakeRelease.resolve();
		await inbox.session.waitForIdle();

		expect(second).toBe("organise_running");
		expect(
			inbox.manager.getEntries().filter(entry => "customType" in entry && entry.customType === TOPIC_ORGANISE_TYPE),
		).toHaveLength(0);
		expect(inbox.session.agent.peekFollowUpQueue()).toHaveLength(1);
	});
});

describe("Parked batches and first-turn admission (#12 review)", () => {
	function conversation(manager: SessionManager): void {
		manager.appendMessage({ role: "user", content: "first question", timestamp: 1 });
		manager.appendMessage(createAssistantMessage("first answer"));
		manager.appendMessage({ role: "user", content: "second question", timestamp: 2 });
		manager.appendMessage(createAssistantMessage("second answer"));
	}

	function rows(manager: SessionManager, customType: string): unknown[] {
		return manager.getEntries().filter(entry => "customType" in entry && entry.customType === customType);
	}

	it("drops an Organise an IRC wake parked when /tree replaces the transcript: the wake never puts it back", async () => {
		const arrived = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const woke = Promise.withResolvers<void>();
		const wakeRelease = Promise.withResolvers<void>();
		const inbox = await openInbox(async context => {
			const text = lastText(context);
			if (text === "long question") {
				arrived.resolve();
				await release.promise;
				return { content: ["the full answer"] };
			}
			if (text.includes("ping")) {
				woke.resolve();
				await wakeRelease.promise;
				return { content: ["pong"] };
			}
			return { content: [text.includes(ORGANISE_MARK) ? "organised" : "ok"] };
		}, conversation);
		const firstAnswer = inbox.manager
			.getEntries()
			.find(entry => entry.type === "message" && textOf(entry.message as Message) === "first answer")!;
		const running = inbox.session.prompt("long question");
		await arrived.promise;
		expect(await inbox.host.organise({ by: "owner" })).toEqual({ queued: true });
		// The user stops the turn: the queued Organise is held for an explicit resume.
		await inbox.session.abort({ reason: USER_INTERRUPT_LABEL });
		release.resolve();
		await running.catch(() => {});
		await inbox.session.waitForIdle();
		expect(inbox.session.agent.peekFollowUpQueue()).toHaveLength(1);
		// An IRC wake parks the held follow-up outside the agent queue for its own turn.
		await inbox.session.deliverIrcMessage({
			id: "m1",
			from: "peer",
			to: "me",
			body: "ping",
			ts: Date.now(),
		} as IrcMessage);
		await woke.promise;
		expect(inbox.session.agent.peekFollowUpQueue()).toHaveLength(0);
		// /tree replaces the transcript while the Organise is parked; the wake then ends and restores its batch.
		const navigating = inbox.session.navigateTree(firstAnswer.id);
		wakeRelease.resolve();
		expect((await navigating).cancelled).toBe(false);
		await inbox.session.waitForIdle();
		expect(inbox.session.agent.peekFollowUpQueue()).toHaveLength(0);
		await inbox.session.prompt("next question");
		await inbox.session.waitForIdle();

		expect(inbox.requests.some(request => request.map(textOf).join("\n").includes(ORGANISE_MARK))).toBe(false);
		expect(rows(inbox.manager, TOPIC_ORGANISE_TYPE)).toEqual([]);
	});

	it("releases an aborted Organise's slot and its wait at abort, not when the prompt ahead of it settles", async () => {
		const inbox = await openInbox(context => ({
			content: [lastText(context).includes(ORGANISE_MARK) ? "organised" : "ok"],
		}));
		const box = sessionInbox(inbox.session);
		// An earlier prompt is held in setup, before its run opens: the Organise turn waits for it.
		const gate = Promise.withResolvers<void>();
		const agentPrompt = inbox.session.agent.prompt.bind(inbox.session.agent) as (...args: unknown[]) => Promise<void>;
		vi.spyOn(inbox.session.agent, "prompt").mockImplementationOnce((async (...args: unknown[]) => {
			await gate.promise;
			return agentPrompt(...args);
		}) as typeof inbox.session.agent.prompt);
		const question = inbox.session.prompt("a question").catch(() => false);
		await Bun.sleep(0);
		expect(inbox.session.hasAdmittedSubmission).toBe(true);
		const organised = box
			.queueTurn(box.capture(), TOPIC_ORGANISE_TYPE, () => ({ content: ORGANISE_MARK, details: { by: "owner" } }))
			.then(
				result => result,
				(error: Error) => error,
			);
		expect(box.turnPending(TOPIC_ORGANISE_TYPE)).toBe(true);
		const replaced = inbox.session.newSession();
		// Settled at abort, while the prompt ahead of it is still held.
		const outcome = await Promise.race([organised, Bun.sleep(500).then(() => "still waiting")]);
		const pending = box.turnPending(TOPIC_ORGANISE_TYPE);
		gate.resolve();
		await replaced;
		await question;
		await inbox.session.waitForIdle();

		expect(outcome).toBeInstanceOf(StaleTranscriptError);
		expect(pending).toBe(false);
		expect(inbox.requests.some(request => request.map(textOf).join("\n").includes(ORGANISE_MARK))).toBe(false);
	});
});

describe("Inbox epochs: bound work is aborted when the transcript is replaced (#11 review)", () => {
	function conversation(manager: SessionManager): void {
		manager.appendMessage({ role: "user", content: "OLD-INBOX-EXCERPT first question", timestamp: 1 });
		manager.appendMessage(createAssistantMessage("first answer"));
		manager.appendMessage({ role: "user", content: "second question", timestamp: 2 });
		manager.appendMessage(createAssistantMessage("second answer"));
	}

	function rows(manager: SessionManager, customType: string): unknown[] {
		return manager.getEntries().filter(entry => "customType" in entry && entry.customType === customType);
	}

	it("drops an Organise queued behind a running turn when /tree moves the branch: never sent, never saved", async () => {
		const arrived = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const inbox = await openInbox(async context => {
			const text = lastText(context);
			if (text === "long question") {
				arrived.resolve();
				await release.promise;
				return { content: ["the full answer"] };
			}
			return { content: [text.includes(ORGANISE_MARK) ? "organised" : "ok"] };
		}, conversation);
		const firstAnswer = inbox.manager
			.getEntries()
			.find(entry => entry.type === "message" && textOf(entry.message as Message) === "first answer")!;
		const running = inbox.session.prompt("long question");
		await arrived.promise;
		expect(await inbox.host.organise({ by: "owner" })).toEqual({ queued: true });
		// While the Organise waits in the queue, /tree moves the branch.
		const navigating = inbox.session.navigateTree(firstAnswer.id);
		release.resolve();
		await running;
		expect((await navigating).cancelled).toBe(false);
		await inbox.session.waitForIdle();

		expect(inbox.requests.some(request => request.map(textOf).join("\n").includes(ORGANISE_MARK))).toBe(false);
		expect(rows(inbox.manager, TOPIC_ORGANISE_TYPE)).toEqual([]);
		expect(await inbox.host.organise({ by: "owner" })).toEqual({ queued: false });
		await inbox.session.waitForIdle();
	});

	it("keeps outgoing work aborted after a /new whose install failed: it never writes into the successor", async () => {
		const inbox = await openInbox(() => ({ content: ["ok"] }), conversation, true);
		await inbox.manager.ensureOnDisk();
		await inbox.manager.flush();
		const box = sessionInbox(inbox.session);
		const outgoing = box.capture();
		const install = inbox.manager.newSession.bind(inbox.manager);
		vi.spyOn(inbox.manager, "newSession").mockImplementationOnce(async (...args) => {
			await install(...args);
			throw new Error("disk full after install");
		});
		await expect(inbox.session.newSession()).rejects.toThrow("disk full after install");

		await expect(box.appendState(outgoing, "probe-outgoing", {})).rejects.toBeInstanceOf(StaleTranscriptError);
		expect(rows(inbox.manager, "probe-outgoing")).toEqual([]);
		expect(box.capture().id).not.toBe(outgoing.id);
	});

	it("keeps outgoing work aborted after a fork whose install failed", async () => {
		const inbox = await openInbox(() => ({ content: ["ok"] }), conversation, true);
		await inbox.manager.ensureOnDisk();
		await inbox.manager.flush();
		const box = sessionInbox(inbox.session);
		const outgoing = box.capture();
		const fork = inbox.manager.fork.bind(inbox.manager);
		vi.spyOn(inbox.manager, "fork").mockImplementationOnce(async () => {
			await fork();
			throw new Error("disk full after install");
		});
		await expect(inbox.session.fork()).rejects.toThrow("disk full after install");

		await expect(box.appendState(outgoing, "probe-outgoing", {})).rejects.toBeInstanceOf(StaleTranscriptError);
		expect(rows(inbox.manager, "probe-outgoing")).toEqual([]);
	});

	it("reopens a suggestion whose seed was cancelled after accept; a repeat accept starts a fresh, linked topic", async () => {
		const summaryAsked = Promise.withResolvers<void>();
		const summary = Promise.withResolvers<string | undefined>();
		let held = true;
		const summarize = vi.fn<SeedSummarizer>(async () => {
			if (!held) return "SUMMARY";
			summaryAsked.resolve();
			return summary.promise;
		});
		const ids: string[] = [];
		const inbox = await openInbox(
			() => ({ content: ["ok"] }),
			manager => {
				for (let n = 1; n <= 32; n++) {
					ids.push(
						n % 2
							? manager.appendMessage({ role: "user", content: `SEED-TEXT ${n}`, timestamp: n })
							: manager.appendMessage(createAssistantMessage(`SEED-TEXT ${n}`)),
					);
				}
			},
			true,
			summarize,
		);
		await inbox.manager.ensureOnDisk();
		await inbox.manager.flush();
		const firstTurns = vi.spyOn(executorModule, "runSubprocess").mockImplementation(async options => {
			options.onFirstPromptAdmitted?.();
			return {
				index: options.index,
				id: options.id,
				agent: options.agent.name,
				agentSource: options.agent.source,
				task: options.task,
				exitCode: 0,
				output: "done",
				stderr: "",
				truncated: false,
				durationMs: 1,
				tokens: 0,
				requests: 0,
			};
		});
		const suggested = Promise.withResolvers<void>();
		inbox.host.onChange(change => {
			if (change.reason === "suggested") suggested.resolve();
		});
		const { id } = inbox.host.suggest({ title: "Long", reason: "r", messageIds: ids });
		await suggested.promise;

		const first = await inbox.host.acceptSuggestion({ suggestionId: id, by: "owner" });
		await summaryAsked.promise;
		expect(await inbox.session.switchSession(inbox.manager.getSessionFile()!)).toBe(true);
		summary.resolve("LATE SUMMARY");
		await inbox.jobs.getJob(`${first.topicId}-t1`)?.promise.catch(() => {});
		await Bun.sleep(0);
		expect(firstTurns).not.toHaveBeenCalled();
		expect(rows(inbox.manager, "topic-suggestion-state")).toEqual([]);
		expect(inbox.host.suggestions().map(open => open.id)).toEqual([id]);

		held = false;
		const retried = await inbox.host.acceptSuggestion({ suggestionId: id, by: "owner" });
		expect(retried.alreadyAccepted).toBe(false);
		expect(retried.topicId).not.toBe(first.topicId);
		await inbox.jobs.getJob(`${retried.topicId}-t1`)?.promise.catch(() => {});
		await Bun.sleep(0);
		expect(firstTurns).toHaveBeenCalledTimes(1);
		expect(rows(inbox.manager, "topic-suggestion-state")).toMatchObject([
			{ data: { id, state: "accepted", topicId: retried.topicId } },
		]);
	});

	it("refuses at once a bound write a transition's own callback awaits: no deadlock", async () => {
		const inbox = await openInbox(() => ({ content: ["ok"] }), conversation, true);
		await inbox.manager.ensureOnDisk();
		await inbox.manager.flush();
		const box = sessionInbox(inbox.session);
		const original = inbox.manager.getSessionFile()!;
		const bound = box.capture();
		const elsewhere = path.join(path.dirname(original), "elsewhere");
		fs.mkdirSync(elsewhere, { recursive: true });
		const target = SessionManager.create(elsewhere, path.dirname(original));
		target.appendMessage({ role: "user", content: "target", timestamp: 3 });
		await target.ensureOnDisk();
		await target.flush();
		let outcome: unknown;
		const switched = await inbox.session
			.switchSession(target.getSessionFile()!, {
				onCwdChange: async () => {
					outcome = await box.appendState(bound, "probe-in-callback", {}).then(
						() => "written",
						(error: Error) => error,
					);
					return false;
				},
			})
			.catch(() => false);

		expect(switched).toBe(false);
		expect(outcome).toBeInstanceOf(StaleTranscriptError);
		expect(rows(inbox.manager, "probe-in-callback")).toEqual([]);
	});
});

describe("Transcript identity across every replacement (#9 review)", () => {
	function conversation(manager: SessionManager): void {
		manager.appendMessage({ role: "user", content: "OLD-INBOX-EXCERPT first question", timestamp: 1 });
		manager.appendMessage(createAssistantMessage("first answer"));
		manager.appendMessage({ role: "user", content: "second question", timestamp: 2 });
		manager.appendMessage(createAssistantMessage("second answer"));
	}

	async function persistedInbox(respond: Parameters<typeof openInbox>[0] = () => ({ content: ["ok"] })) {
		const inbox = await openInbox(respond, conversation, true);
		await inbox.manager.ensureOnDisk();
		await inbox.manager.flush();
		return inbox;
	}

	function stateRows(manager: SessionManager, customType: string): unknown[] {
		return manager.getEntries().filter(entry => entry.type === "custom" && entry.customType === customType);
	}

	it("aborts work bound before a switch, refuses capture during it at once, and starts a new epoch after its rollback", async () => {
		const inbox = await persistedInbox();
		const box = sessionInbox(inbox.session);
		const original = inbox.manager.getSessionFile()!;
		const early = box.capture();
		// A session recorded in another directory: the switch asks to change cwd, and that is refused.
		const elsewhere = path.join(path.dirname(original), "elsewhere");
		fs.mkdirSync(elsewhere, { recursive: true });
		const target = SessionManager.create(elsewhere, path.dirname(original));
		target.appendMessage({ role: "user", content: "target", timestamp: 3 });
		await target.ensureOnDisk();
		await target.flush();
		const asked = Promise.withResolvers<void>();
		const decide = Promise.withResolvers<boolean>();
		const switched = inbox.session.switchSession(target.getSessionFile()!, {
			onCwdChange: async () => {
				asked.resolve();
				return decide.promise;
			},
		});
		await asked.promise;
		expect(inbox.session.isSessionTransitioning).toBe(true);
		// The switch began: work bound before it is aborted, and nothing can bind until it settles.
		expect(early.signal.aborted).toBe(true);
		expect(() => box.capture()).toThrow(StaleTranscriptError);
		const earlyWrite = box.appendState(early, "probe-early", {}).then(
			() => "written",
			(error: Error) => error,
		);
		decide.resolve(false);
		await switched.catch(() => false);

		expect(inbox.manager.getSessionFile()).toBe(original);
		expect(await earlyWrite).toBeInstanceOf(StaleTranscriptError);
		expect(stateRows(inbox.manager, "probe-early")).toEqual([]);
		// Rolled back to the original transcript: a new epoch, never the aborted one revived.
		const after = box.capture();
		expect(after.id).not.toBe(early.id);
		await box.appendState(after, "probe-after", {});
		expect(stateRows(inbox.manager, "probe-after")).toHaveLength(1);
	});

	it("refuses an Organise held in deferred loading across a /btw promotion", async () => {
		const inbox = await persistedInbox(context => ({
			content: [lastText(context).includes(ORGANISE_MARK) ? "organised" : "ok"],
		}));
		const gate = Promise.withResolvers<void>();
		let pending = true;
		Object.defineProperty(inbox.session, "hasPendingDeferredExtensions", { configurable: true, get: () => pending });
		vi.spyOn(inbox.session, "loadDeferredExtensions").mockImplementation(async () => {
			await gate.promise;
			pending = false;
		});
		const organised = inbox.host.organise({ by: "owner" }).then(
			result => result,
			(error: Error) => error,
		);
		await Bun.sleep(0);
		const promoted = await inbox.session.branchFromBtw(
			"side question",
			createAssistantMessage("side answer"),
			inbox.manager.getLeafId()!,
			inbox.manager.getSessionId(),
		);
		expect(promoted.cancelled).toBe(false);
		gate.resolve();
		expect(await organised).toBeInstanceOf(StaleTranscriptError);
		await inbox.session.waitForIdle();
		expect(inbox.requests.some(request => request.map(textOf).join("\n").includes(ORGANISE_MARK))).toBe(false);
	});

	it("refuses an atomic write pending across /tree navigation: nothing reaches the file", async () => {
		const inbox = await persistedInbox();
		const box = sessionInbox(inbox.session);
		const file = inbox.manager.getSessionFile()!;
		const firstAnswer = inbox.manager
			.getEntries()
			.find(entry => entry.type === "message" && textOf(entry.message as Message) === "first answer")!;
		const reached = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const ensureOnDisk = inbox.manager.ensureOnDisk.bind(inbox.manager);
		vi.spyOn(inbox.manager, "ensureOnDisk").mockImplementationOnce(async () => {
			reached.resolve();
			await release.promise;
			await ensureOnDisk();
		});
		const write = box.appendState(box.capture(), "probe-tree", {}).then(
			() => "written",
			(error: Error) => error,
		);
		await reached.promise;
		const navigated = await inbox.session.navigateTree(firstAnswer.id);
		expect(navigated.cancelled).toBe(false);
		release.resolve();
		expect(await write).toBeInstanceOf(StaleTranscriptError);
		await inbox.manager.flush();
		expect(await Bun.file(file).text()).not.toContain("probe-tree");
	});

	it("keeps the transcript identity on a fork that does nothing (no session file)", async () => {
		const inbox = await openInbox(() => ({ content: ["ok"] }), conversation);
		const epoch = inbox.session.captureInboxEpoch();
		expect(await inbox.session.fork()).toBe(false);
		expect(epoch.signal.aborted).toBe(false);
		expect(inbox.session.captureInboxEpoch().id).toBe(epoch.id);
	});
	it("no longer offers or accepts a suggestion whose messages /clear removed", async () => {
		let ask = "";
		const inbox = await openInbox(
			() => ({ content: ["ok"] }),
			manager => {
				ask = manager.appendMessage({ role: "user", content: "Plan the offsite", timestamp: 1 });
				manager.appendMessage(createAssistantMessage("sure"));
			},
		);
		const suggested = Promise.withResolvers<void>();
		inbox.host.onChange(change => {
			if (change.reason === "suggested") suggested.resolve();
		});
		const { id } = inbox.host.suggest({ title: "Offsite", reason: "r", messageIds: [ask] });
		await suggested.promise;
		expect(inbox.host.suggestions().map(open => open.id)).toEqual([id]);

		expect(await inbox.session.resetSessionContext()).toBeDefined();
		expect(inbox.host.suggestions()).toEqual([]);
		expect(inbox.host.suggestionBoard().open).toEqual([]);
		expect(await codeOf(() => inbox.host.acceptSuggestion({ suggestionId: id, by: "owner" }))).toBe(
			"suggestion_not_found",
		);
	});

	it("cancels a seeded topic whose summary settles after a same-file reload: no first turn, no verbatim fallback", async () => {
		const summaryAsked = Promise.withResolvers<void>();
		const summary = Promise.withResolvers<string | undefined>();
		const summarize = vi.fn<SeedSummarizer>(() => {
			summaryAsked.resolve();
			return summary.promise;
		});
		const ids: string[] = [];
		const inbox = await openInbox(
			() => ({ content: ["ok"] }),
			manager => {
				for (let n = 1; n <= 32; n++) {
					ids.push(
						n % 2
							? manager.appendMessage({ role: "user", content: `SEED-TEXT ${n}`, timestamp: n })
							: manager.appendMessage(createAssistantMessage(`SEED-TEXT ${n}`)),
					);
				}
			},
			true,
			summarize,
		);
		await inbox.manager.ensureOnDisk();
		await inbox.manager.flush();
		const firstTurn = vi.spyOn(executorModule, "runSubprocess");
		const { id } = await inbox.host.create({ title: "Long", message: "Go on.", seedMessageIds: ids });
		await summaryAsked.promise;
		expect(await inbox.session.switchSession(inbox.manager.getSessionFile()!)).toBe(true);
		summary.resolve("LATE SUMMARY");
		await inbox.jobs.getJob(`${id}-t1`)?.promise.catch(() => {});

		expect(firstTurn).not.toHaveBeenCalled();
	});
});

describe("Inbox generation checked at the point of effect (#8 review)", () => {
	const OLD = "OLD-INBOX-EXCERPT";
	function withOldExcerpt(manager: SessionManager): void {
		manager.appendMessage({ role: "user", content: `${OLD} about the Diokan deal`, timestamp: 1 });
		manager.appendMessage(createAssistantMessage("noted"));
	}

	it("refuses an Organise held in the deferred extension loader across /new: nothing of the old Inbox is sent or saved", async () => {
		const inbox = await openInbox(
			context => ({ content: [lastText(context).includes(ORGANISE_MARK) ? "organised" : "ok"] }),
			withOldExcerpt,
		);
		const gate = Promise.withResolvers<void>();
		let pending = true;
		Object.defineProperty(inbox.session, "hasPendingDeferredExtensions", { configurable: true, get: () => pending });
		vi.spyOn(inbox.session, "loadDeferredExtensions").mockImplementation(async () => {
			await gate.promise;
			pending = false;
		});
		const organised = inbox.host.organise({ by: "owner" }).then(
			result => result,
			(error: Error) => error,
		);
		await Bun.sleep(0);
		const before = inbox.requests.length;
		await inbox.session.newSession();
		gate.resolve();
		const outcome = await organised;
		await inbox.session.waitForIdle();

		expect(outcome).toBeInstanceOf(StaleTranscriptError);
		expect(JSON.stringify(inbox.manager.getEntries())).not.toContain(OLD);
		for (const request of inbox.requests.slice(before)) expect(request.map(textOf).join("\n")).not.toContain(OLD);
	});

	it("refuses a seeded topics_new whose spawn write returns after /new: no summary request, no old seed text saved", async () => {
		const summarize = vi.fn<SeedSummarizer>(async () => "SUMMARY");
		const ids: string[] = [];
		const inbox = await openInbox(
			() => ({ content: ["ok"] }),
			manager => {
				for (let n = 1; n <= 32; n++) {
					ids.push(
						n % 2
							? manager.appendMessage({ role: "user", content: `OLD-SEED-TEXT ${n}`, timestamp: n })
							: manager.appendMessage(createAssistantMessage(`OLD-SEED-TEXT ${n}`)),
					);
				}
			},
			true,
			summarize,
		);
		await inbox.manager.ensureOnDisk();
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async (options: ExecutorOptions) => {
			options.onFirstPromptAdmitted?.();
			return {
				index: options.index,
				id: options.id,
				agent: options.agent.name,
				agentSource: options.agent.source,
				task: options.task,
				exitCode: 0,
				output: "done",
				stderr: "",
				truncated: false,
				durationMs: 1,
				tokens: 0,
				requests: 0,
			};
		});
		// Hold the spawn's durable write (its lifecycle entry) until /new has completed.
		const flushed = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const flush = inbox.manager.flush.bind(inbox.manager);
		let held = false;
		vi.spyOn(inbox.manager, "flush").mockImplementation(async () => {
			await flush();
			const spawnWrite = inbox.manager
				.getEntries()
				.some(entry => entry.type === "custom" && entry.customType === "vibe-session-lifecycle");
			if (held || !spawnWrite) return;
			held = true;
			flushed.resolve();
			await release.promise;
		});
		const created = inbox.host.create({ title: "Diokan", message: "Go on.", seedMessageIds: ids }).then(
			result => result,
			(error: Error) => error,
		);
		await flushed.promise;
		await inbox.session.newSession();
		release.resolve();
		const outcome = await created;
		// The topic's first turn runs in the background; give it a moment to (not) ask for a summary.
		await Bun.sleep(20);

		expect(outcome).toBeInstanceOf(StaleTranscriptError);
		expect(summarize).not.toHaveBeenCalled();
		expect(JSON.stringify(inbox.manager.getEntries())).not.toContain("OLD-SEED-TEXT");
	});

	it("builds Organise's index from the conversation after /clear only", async () => {
		const inbox = await openInbox(
			context => ({ content: [lastText(context).includes(ORGANISE_MARK) ? "organised" : "ok"] }),
			manager => {
				manager.appendMessage({ role: "user", content: "PRE-CLEAR-TEXT secret plans", timestamp: 1 });
				manager.appendMessage(createAssistantMessage("PRE-CLEAR-TEXT acknowledged"));
			},
		);
		expect(await inbox.session.resetSessionContext()).toBeDefined();
		expect(await inbox.host.organise({ by: "owner" })).toEqual({ queued: false });
		await inbox.session.waitForIdle();
		const sent = inbox.requests.at(-1)!.map(textOf).join("\n");
		expect(sent).toContain(ORGANISE_MARK);
		expect(sent).not.toContain("PRE-CLEAR-TEXT");
	});

	it("refuses a state write bound before a same-file reopen, under the write lock: nothing reaches disk", async () => {
		const inbox = await openInbox(() => ({ content: ["ok"] }), withOldExcerpt, true);
		await inbox.manager.ensureOnDisk();
		await inbox.manager.flush();
		const file = inbox.manager.getSessionFile()!;
		const box = sessionInbox(inbox.session);
		const binding = box.capture();
		expect(await inbox.session.switchSession(file)).toBe(true);

		await expect(box.appendState(binding, "probe-stale-state", { stale: true })).rejects.toBeInstanceOf(
			StaleTranscriptError,
		);
		await inbox.manager.flush();
		expect(await Bun.file(file).text()).not.toContain("probe-stale-state");
	});
});

describe("Inbox binding (#7 review)", () => {
	it("refuses an Organise whose Inbox was replaced while it waited behind a prompt in setup", async () => {
		const inbox = await openInbox(
			context => ({ content: [lastText(context).includes(ORGANISE_MARK) ? "organised" : "the answer"] }),
			manager => {
				manager.appendMessage({ role: "user", content: "OLD-INBOX-EXCERPT about the Diokan deal", timestamp: 1 });
				manager.appendMessage(createAssistantMessage("noted"));
			},
		);
		// An earlier prompt is held in setup, before its run opens.
		const gate = Promise.withResolvers<void>();
		const agentPrompt = inbox.session.agent.prompt.bind(inbox.session.agent) as (...args: unknown[]) => Promise<void>;
		vi.spyOn(inbox.session.agent, "prompt").mockImplementationOnce((async (...args: unknown[]) => {
			await gate.promise;
			return agentPrompt(...args);
		}) as typeof inbox.session.agent.prompt);
		const question = inbox.session.prompt("a question").catch(() => false);
		await Bun.sleep(0);
		expect(inbox.session.hasAdmittedSubmission).toBe(true);
		const organised = inbox.host.organise({ by: "owner" }).then(
			result => result,
			(error: Error) => error,
		);
		const before = inbox.requests.length;
		const replaced = inbox.session.newSession();
		await Bun.sleep(0);
		gate.resolve();
		await replaced;
		await question;
		const outcome = await organised;
		await inbox.session.waitForIdle();

		expect(outcome).toBeInstanceOf(Error);
		expect(JSON.stringify(inbox.manager.getEntries())).not.toContain("OLD-INBOX-EXCERPT");
		for (const request of inbox.requests.slice(before)) {
			expect(request.map(textOf).join("\n")).not.toContain("OLD-INBOX-EXCERPT");
		}
	});

	it("rejects a row the session never saves at once, instead of leaving it pending", async () => {
		const inbox = await openInbox(() => ({ content: ["ok"] }));
		const recorded = inbox.session.recordWithoutTurn({
			role: "custom",
			customType: "vibe-mode-context",
			content: "vibe",
			display: false,
			attribution: "agent",
			timestamp: Date.now(),
		});
		const settled = await Promise.race([
			recorded.then(
				() => "resolved",
				(error: Error) => error.message,
			),
			Bun.sleep(50).then(() => "pending"),
		]);
		expect(settled).toContain("never saved");
	});
});

describe("Inbox writes (#6 review)", () => {
	it("records a row in a new session's file, creating it, before reporting it saved", async () => {
		const inbox = await openInbox(
			() => ({ content: ["ok"] }),
			manager => {
				manager.appendMessage({ role: "user", content: "no answer yet", timestamp: 1 });
			},
			true,
		);
		const file = inbox.manager.getSessionFile()!;
		expect(await Bun.file(file).exists()).toBe(false);
		await inbox.session.recordWithoutTurn({
			role: "custom",
			customType: "note-durable",
			content: "DURABLE-NOTE",
			display: true,
			attribution: "agent",
			timestamp: Date.now(),
		});
		expect(await Bun.file(file).exists()).toBe(true);
		expect(await Bun.file(file).text()).toContain("DURABLE-NOTE");
	});

	it("retries a dismissal whose write failed instead of reporting it already dismissed", async () => {
		let ask = "";
		const inbox = await openInbox(
			() => ({ content: ["ok"] }),
			manager => {
				ask = manager.appendMessage({ role: "user", content: "Plan the offsite", timestamp: 1 });
			},
			true,
		);
		const suggested = Promise.withResolvers<void>();
		const changes: string[] = [];
		inbox.host.onChange(change => {
			changes.push(change.reason);
			if (change.reason === "suggested") suggested.resolve();
		});
		const { id } = inbox.host.suggest({ title: "Offsite", reason: "r", messageIds: [ask] });
		await suggested.promise;

		const flush = vi.spyOn(inbox.manager, "flush").mockRejectedValueOnce(new Error("disk full"));
		await expect(inbox.host.dismissSuggestion({ suggestionId: id, by: "owner" })).rejects.toThrow("disk full");
		expect(changes).not.toContain("suggestion-dismissed");
		flush.mockRestore();

		expect(await inbox.host.dismissSuggestion({ suggestionId: id, by: "owner" })).toEqual({ alreadyDismissed: false });
		expect(changes).toContain("suggestion-dismissed");
		const states = inbox.manager
			.getEntries()
			.filter(entry => entry.type === "custom" && entry.customType === "topic-suggestion-state");
		expect(states).toHaveLength(1);
	});

	it("lets a prompt still in setup go first: Organise queues behind it and the prompt does not fail", async () => {
		const inbox = await openInbox(context => ({
			content: [lastText(context).includes(ORGANISE_MARK) ? "organised" : "the answer"],
		}));
		const question = inbox.session.prompt("a question");
		expect(inbox.session.hasAdmittedSubmission).toBe(true);
		const organised = inbox.host.organise({ by: "owner" });
		await expect(question).resolves.toBe(true);
		expect(await organised).toEqual({ queued: true });
		await inbox.session.waitForIdle();
		expect(inbox.requests).toHaveLength(2);
		expect(inbox.requests[0]!.map(textOf).join("\n")).toContain("a question");
		expect(inbox.requests[1]!.map(textOf).at(-1)).toContain(ORGANISE_MARK);
	});

	it("fails Organise when its turn cannot start (no model) instead of reporting success", async () => {
		const inbox = await openInbox(() => ({ content: ["unused"] }));
		inbox.session.agent.state.model = undefined as never;
		await expect(inbox.host.organise({ by: "owner" })).rejects.toThrow();
		expect(inbox.requests).toHaveLength(0);
		// Nothing is left pending: a later Organise is not refused as running.
		await expect(inbox.host.organise({ by: "owner" })).rejects.not.toMatchObject({ code: "organise_running" });
	});

	it("rejects a note whose row is dropped while a transition has the session detached and it is disposed", async () => {
		const inbox = await openInbox(() => ({ content: ["unused"] }));
		const setup = Promise.withResolvers<void>();
		const failWake = Promise.withResolvers<never>();
		vi.spyOn(inbox.session.agent, "prompt").mockImplementationOnce(() => {
			setup.resolve();
			return failWake.promise;
		});
		await inbox.session.deliverIrcMessage({ id: "m1", from: "Helper", to: OWNER, body: "ping", ts: Date.now() });
		await setup.promise;
		const recorded = inbox.session.recordWithoutTurn({
			role: "custom",
			customType: "note-race",
			content: "NOTE-TEXT",
			display: true,
			attribution: "agent",
			timestamp: Date.now(),
		});
		const settled = recorded.then(
			() => "saved",
			(error: Error) => error.message,
		);
		// Disposal starts while the wake is still in setup; a transition then detaches the session and aborts the wake.
		const disposed = inbox.session.dispose({ drainTimeoutMs: 50 });
		const transition = inbox.session.newSession().catch(() => false);
		sessions.splice(sessions.indexOf(inbox.session), 1);
		failWake.reject(new Error("wake setup failed"));
		await disposed;
		await transition;
		expect(await settled).toBe("The session was disposed before the row was recorded.");
	});
});

describe("Inbox suggestions and Organise", () => {
	it("saves a suggestion from the Inbox model after its turn, refuses unknown ids, and boards it", async () => {
		let ask = "";
		let suggestCall = 0;
		const inbox = await openInbox(
			context => {
				const text = lastText(context);
				if (text === "please organise this") {
					return {
						content: [
							{
								type: "toolCall",
								name: "topic_suggest",
								arguments: { title: "Diokan deal", reason: "A separate deal.", message_ids: [ask] },
							},
							{
								type: "toolCall",
								name: "topic_suggest",
								arguments: { title: "Ghost", reason: "r", message_ids: ["no-such-entry"] },
							},
						],
					};
				}
				suggestCall++;
				return { content: [suggestCall === 1 ? "done" : "ok"] };
			},
			manager => {
				ask = manager.appendMessage({ role: "user", content: "Let's work on the Diokan deal", timestamp: 1 });
				manager.appendMessage(createAssistantMessage("Sure, what first?"));
			},
		);
		const saved = Promise.withResolvers<void>();
		inbox.host.onChange(change => {
			if (change.reason === "suggested") saved.resolve();
		});

		await inbox.session.prompt("please organise this");
		await inbox.session.waitForIdle();
		await saved.promise;

		const entries = inbox.manager.getEntries();
		const suggestionAt = entries.findIndex(
			entry => entry.type === "custom_message" && entry.customType === TOPIC_SUGGESTION_TYPE,
		);
		const doneAt = entries.findIndex(
			entry => entry.type === "message" && entry.message.role === "assistant" && textOf(entry.message) === "done",
		);
		expect(doneAt).toBeGreaterThan(-1);
		expect(suggestionAt).toBeGreaterThan(doneAt);
		expect(entries[suggestionAt]).toMatchObject({
			display: true,
			details: { title: "Diokan deal", reason: "A separate deal.", messageIds: [ask] },
		} satisfies Partial<CustomMessageEntry>);
		expect(inbox.host.suggestions().map(suggestion => suggestion.title)).toEqual(["Diokan deal"]);

		const results = entries.flatMap(entry =>
			entry.type === "message" && entry.message.role === "toolResult" ? [entry.message] : [],
		);
		expect(results).toHaveLength(2);
		expect(results[1]).toMatchObject({ isError: true });
		expect(textOf(results[1]!)).toContain("invalid_message_ids");

		await inbox.session.prompt("anything else?");
		await inbox.session.waitForIdle();
		const board = inbox.requests.at(-1)!.map(textOf).find(text => text.includes("<topic-board>"));
		expect(board).toContain("Diokan deal (suggestion sg_");
	});

	it("runs Organise as one hidden Inbox turn when idle and refuses a second while it runs", async () => {
		const arrived = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const again = Promise.withResolvers<void>();
		let organiseTurns = 0;
		const inbox = await openInbox(async context => {
			if (lastText(context).includes(ORGANISE_MARK)) {
				if (organiseTurns++ === 0) {
					arrived.resolve();
					await release.promise;
				} else {
					again.resolve();
				}
			}
			return { content: ["organised"] };
		});

		expect(await inbox.host.organise({ by: "owner" })).toEqual({ queued: false });
		await arrived.promise;
		expect(await codeOf(() => inbox.host.organise({ by: "owner" }))).toBe("organise_running");

		release.resolve();
		await inbox.session.waitForIdle();
		const instruction = inbox.manager
			.getEntries()
			.find(entry => entry.type === "custom_message" && entry.customType === TOPIC_ORGANISE_TYPE);
		expect(instruction).toMatchObject({ display: false, details: { by: "owner" } });

		expect(await inbox.host.organise({ by: "owner" })).toEqual({ queued: false });
		await again.promise;
		await inbox.session.waitForIdle();
		expect(organiseTurns).toBe(2);
	});

	it("queues Organise behind a running turn without interrupting it", async () => {
		const arrived = Promise.withResolvers<void>();
		const release = Promise.withResolvers<void>();
		const inbox = await openInbox(async context => {
			const text = lastText(context);
			if (text === "long question") {
				arrived.resolve();
				await release.promise;
				return { content: ["the full answer"] };
			}
			return { content: [text.includes(ORGANISE_MARK) ? "organised" : "?"] };
		});

		const running = inbox.session.prompt("long question");
		await arrived.promise;
		expect(await inbox.host.organise({ by: "guest Anna" })).toEqual({ queued: true });
		expect(await codeOf(() => inbox.host.organise({ by: "owner" }))).toBe("organise_running");
		expect(inbox.requests).toHaveLength(1);

		release.resolve();
		await running;
		await inbox.session.waitForIdle();

		expect(inbox.requests).toHaveLength(2);
		const second = inbox.requests[1]!.map(textOf);
		expect(second).toContain("the full answer");
		expect(second.at(-1)).toContain(ORGANISE_MARK);
		const answer = inbox.manager
			.getEntries()
			.find(
				entry =>
					entry.type === "message" &&
					entry.message.role === "assistant" &&
					textOf(entry.message) === "the full answer",
			);
		expect(answer).toMatchObject({ message: { stopReason: "stop" } });
		expect(await inbox.host.organise({ by: "owner" })).toEqual({ queued: false });
		await inbox.session.waitForIdle();
	});
});
