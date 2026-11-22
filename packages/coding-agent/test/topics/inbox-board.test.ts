/**
 * Contract: the Inbox (main session) sends its topic board as a hidden context
 * message ahead of a turn, after everything the previous request sent, and only
 * when the board changed since the last one in its history. The board never
 * enters the system prompt, an unchanged board adds nothing, and the board is
 * persisted at its slot, so a resumed process sends the same request prefix.
 * Without topics enabled, or with no topics, no board is sent.
 */
import { afterEach, describe, expect, it } from "bun:test";
import * as path from "node:path";
import { Agent } from "@oh-my-pi/pi-agent-core";
import type { Message } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { TopicsHost } from "@oh-my-pi/pi-coding-agent/topics/host";
import type { WorkerSummary } from "@oh-my-pi/pi-coding-agent/vibe/runtime";
import { TempDir } from "@oh-my-pi/pi-utils";

const T0 = Date.UTC(2026, 10, 19, 7, 5, 30);
const SYSTEM_PROMPT = "Stable system prompt";

let temp: TempDir;
const sessions: AgentSession[] = [];
let auth: AuthStorage | undefined;

afterEach(async () => {
	for (const session of sessions.splice(0)) await session.dispose();
	auth?.close();
	auth = undefined;
	temp?.removeSync();
});

interface Request {
	systemPrompt: string[];
	messages: Message[];
}

interface Process {
	session: AgentSession;
	manager: SessionManager;
	requests: Request[];
}

function topic(id: string, overrides: Partial<WorkerSummary> = {}): WorkerSummary {
	return {
		id,
		title: `${id} title`,
		state: "idle",
		turns: 1,
		queued: 0,
		createdAt: T0,
		lastActivityAt: T0,
		...overrides,
	};
}

/** A topics host listing whatever `topics` holds when asked. */
function hostOf(topics: WorkerSummary[]): TopicsHost {
	return {
		list: () => topics,
		suggestionBoard: () => ({ open: [], dismissed: [] }),
		restore: async () => 0,
		dispose: () => {},
	} as unknown as TopicsHost;
}

async function createRegistry(): Promise<ModelRegistry> {
	auth = await AuthStorage.create(":memory:");
	auth.keys.setRuntime("anthropic", "test-key");
	return new ModelRegistry(auth);
}

/** One omp process on `file`; `topics` set = topics enabled with that list. */
async function openProcess(
	file: string,
	sessionDir: string,
	registry: ModelRegistry,
	topics: WorkerSummary[] | undefined,
): Promise<Process> {
	const bundled = getBundledModel("anthropic", "claude-sonnet-4-5");
	if (!bundled) throw new Error("Expected bundled model");
	const manager = await SessionManager.open(file, sessionDir);
	const requests: Request[] = [];
	const mock = createMockModel({ provider: "anthropic", handler: { content: ["OK"] } });
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: {
			model: bundled,
			systemPrompt: [SYSTEM_PROMPT],
			tools: [],
			messages: manager.buildSessionContext().messages,
		},
		convertToLlm,
		streamFn: (model, context, options) => {
			requests.push(structuredClone({ systemPrompt: context.systemPrompt ?? [], messages: context.messages }));
			return mock.stream(model, context, options);
		},
	});
	const session = new AgentSession({
		agent,
		sessionManager: manager,
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: registry,
		createTopicsHost: topics ? () => hostOf(topics) : undefined,
	});
	sessions.push(session);
	return { session, manager, requests };
}

async function seedSession(): Promise<{ file: string; sessionDir: string }> {
	temp = TempDir.createSync("@omp-topic-board-");
	const sessionDir = path.join(temp.path(), "sessions");
	const seed = SessionManager.create(temp.path(), sessionDir);
	seed.appendMessage({ role: "user", content: "earlier", timestamp: T0 });
	await seed.ensureOnDisk();
	await seed.flush();
	const file = seed.getSessionFile();
	if (!file) throw new Error("Expected a persisted session");
	return { file, sessionDir };
}

async function promptOnce(proc: Process, text: string): Promise<Request> {
	await proc.session.prompt(text);
	await proc.session.waitForIdle();
	await proc.manager.flush();
	const last = proc.requests.at(-1);
	if (!last) throw new Error("Expected a provider request");
	return last;
}

function textOf(message: Message): string {
	const content = message.content;
	if (typeof content === "string") return content;
	return content.map(block => ("text" in block ? block.text : "")).join("");
}

/** Indexes of the board messages a request sent. */
function boards(request: Request): number[] {
	return request.messages.flatMap((message, index) => (textOf(message).includes("<topic-board>") ? [index] : []));
}

const prefix = (messages: Message[]) => messages.map(message => ({ role: message.role, content: message.content }));

describe("inbox topic board", () => {
	it("appends the board after the cached prefix only when it changed, never in the system prompt", async () => {
		const { file, sessionDir } = await seedSession();
		const registry = await createRegistry();
		const topics = [topic("Diokan", { state: "running", lastActivity: "Drafting clause 4" })];
		const live = await openProcess(file, sessionDir, registry, topics);

		const first = await promptOnce(live, "one");
		expect(boards(first)).toEqual([first.messages.length - 2]);
		expect(textOf(first.messages[first.messages.length - 2]!)).toContain("(id: Diokan): working");
		expect(textOf(first.messages.at(-1)!)).toBe("one");

		// Unchanged board: the turn adds only its own messages.
		const second = await promptOnce(live, "two");
		expect(prefix(second.messages).slice(0, first.messages.length)).toEqual(prefix(first.messages));
		expect(boards(second)).toEqual(boards(first));
		expect(second.messages.length).toBe(first.messages.length + 2);

		// Changed board: one new board at the end, ahead of the new prompt; the earlier messages are untouched.
		topics[0] = topic("Diokan", { lastActivityAt: T0 + 120_000, lastActivity: "Sent the draft" });
		const third = await promptOnce(live, "three");
		expect(prefix(third.messages).slice(0, second.messages.length)).toEqual(prefix(second.messages));
		expect(boards(third)).toEqual([...boards(second), third.messages.length - 2]);
		expect(textOf(third.messages[third.messages.length - 2]!)).toContain("(id: Diokan): idle");

		for (const request of [first, second, third]) {
			expect(request.systemPrompt).toEqual([SYSTEM_PROMPT]);
			expect(request.systemPrompt.join("\n")).not.toContain("topic-board");
		}

		// The boards are persisted at their slots: a resumed process sends the same prefix, and no new board.
		const resumed = await openProcess(file, sessionDir, registry, topics);
		const fourth = await promptOnce(resumed, "four");
		expect(prefix(fourth.messages).slice(0, third.messages.length)).toEqual(prefix(third.messages));
		expect(boards(fourth)).toEqual(boards(third));
	});

	it("sends no board when topics are disabled, where the same session with topics would", async () => {
		const { file, sessionDir } = await seedSession();
		const registry = await createRegistry();
		const disabled = await openProcess(file, sessionDir, registry, undefined);
		expect(boards(await promptOnce(disabled, "one"))).toEqual([]);
		const enabled = await openProcess(file, sessionDir, registry, [topic("Notes")]);
		const second = await promptOnce(enabled, "two");
		expect(boards(second)).toEqual([second.messages.length - 2]);
	});

	it("sends no board while no topic exists, and one once a topic does", async () => {
		const { file, sessionDir } = await seedSession();
		const topics: WorkerSummary[] = [];
		const proc = await openProcess(file, sessionDir, await createRegistry(), topics);
		expect(boards(await promptOnce(proc, "one"))).toEqual([]);
		topics.push(topic("Notes"));
		const second = await promptOnce(proc, "two");
		expect(boards(second)).toEqual([second.messages.length - 2]);
	});

	it("sends a board supplied with the prompt (a topic's siblings) once until it changes", async () => {
		const { file, sessionDir } = await seedSession();
		const proc = await openProcess(file, sessionDir, await createRegistry(), undefined);
		const sibling = "<topic-board>\n- Notes title (id: Notes): idle\n</topic-board>";
		await proc.session.prompt("one", { topicBoard: sibling });
		await proc.session.waitForIdle();
		await proc.session.prompt("two", { topicBoard: sibling });
		await proc.session.waitForIdle();
		const second = proc.requests.at(-1)!;
		expect(boards(second)).toEqual([second.messages.length - 4]);
		expect(textOf(second.messages[second.messages.length - 4]!)).toBe(sibling);
	});
});
