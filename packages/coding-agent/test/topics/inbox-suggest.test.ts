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
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import type { CustomMessageEntry, SessionEntry } from "@oh-my-pi/pi-coding-agent/session/session-entries";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { createTopicsHost, type TopicsHost } from "@oh-my-pi/pi-coding-agent/topics/host";
import { sessionInbox } from "@oh-my-pi/pi-coding-agent/topics/inbox";
import { TopicSuggestTool } from "@oh-my-pi/pi-coding-agent/topics/suggest-tool";
import { TOPIC_ORGANISE_TYPE, TOPIC_SUGGESTION_TYPE, TopicsError } from "@oh-my-pi/pi-coding-agent/topics/suggestions";
import { resetTopicsRegistryForTests } from "@oh-my-pi/pi-coding-agent/topics/workers";
import { TempDir } from "@oh-my-pi/pi-utils";
import { createAssistantMessage, createInMemoryAuthStorage } from "../helpers/agent-session-setup";

const OWNER = "Main";
const ORGANISE_MARK = "call `topic_suggest` 3–5 times";

const sessions: AgentSession[] = [];
const managers: AsyncJobManager[] = [];
const temps: TempDir[] = [];

afterEach(async () => {
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
}

/** An Inbox session with topics on, its model answering through `respond`; `persist` = a session file on disk. */
async function openInbox(
	respond: (context: Context) => MockResponse | Promise<MockResponse>,
	seed: (manager: SessionManager) => void = () => {},
	persist = false,
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
		getSessionFile: () => null,
		getArtifactsDir: () => null,
		taskDepth: 0,
		enableLsp: false,
	} as unknown as ToolSession;
	let session!: AgentSession;
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
	session = new AgentSession({
		agent,
		sessionManager: manager,
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: new ModelRegistry(auth),
		createTopicsHost: () =>
			createTopicsHost(toolSession, { inbox: sessionInbox(session), summarize: async () => "summary" }),
	});
	sessions.push(session);
	const host = await session.getTopicsHost();
	if (!host) throw new Error("Expected a topics host");
	return { session, manager, requests, host };
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
		expect(await box.appendNote(inbox.manager.getSessionId(), { customType: "note-a", content: "first", details: {} })).toBe(true);
		expect(await box.appendNote(inbox.manager.getSessionId(), { customType: "note-b", content: "second", details: {} })).toBe(true);
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
		const saved = box.appendNote(inbox.manager.getSessionId(), { customType: "note-race", content: "NOTE-TEXT", details: {} });
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
		const saved = box.appendNote(inbox.manager.getSessionId(), { customType: "note-race", content: "NOTE-TEXT", details: {} });
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
		const saved = box.appendNote(inbox.manager.getSessionId(), { customType: "note-race", content: "NOTE-TEXT", details: {} });
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
		expect(await sessionInbox(inbox.session).appendNote(inbox.manager.getSessionId(), { customType: "note-live", content: "x", details: {} })).toBe(
			true,
		);
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

		expect(
			await sessionInbox(inbox.session).appendNote(inbox.manager.getSessionId(), {
				customType: "note-fail",
				content: "x",
				details: {},
			}),
		).toBe(false);
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
