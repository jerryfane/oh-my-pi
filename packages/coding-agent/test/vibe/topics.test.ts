/**
 * Contract: the topics registry keeps owner-facing, permanent workers on the
 * Vibe runtime without disturbing Vibe mode.
 *
 * - A topic's turn result is NOT delivered into the parent conversation (the
 *   owner reads it in the topic), while a Vibe worker's result still is.
 * - Title, close and reopen persist in the parent session and survive a
 *   restart: a closed topic is restored closed (never terminal) and can be
 *   reopened; Vibe rehydration never adopts topic records.
 * - Each topic turn carries a board of the owner's other topics (never
 *   itself); Vibe workers get none. The ids it lists address the topics:
 *   `history://<id>` reads one. A working topic's row stays put through its
 *   turn's progress and moves when the turn settles.
 * - With a topics host, an agent's `agent://<id>` message to a topic runs as a
 *   topic turn: sibling board, counted, labelled with the sender, answered
 *   back to the sender. A busy topic queues it, a closed one refuses it, and a
 *   turn an ask started cannot ask other topics (no chains).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { InternalUrlRouter } from "@oh-my-pi/pi-coding-agent/internal-urls";
import { IrcBus } from "@oh-my-pi/pi-coding-agent/irc/bus";
import { ircRouteCountForTests } from "@oh-my-pi/pi-coding-agent/irc/routes";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { ExecutorOptions } from "@oh-my-pi/pi-coding-agent/task/executor";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { formatTopicBoard, topicBoardMessage } from "@oh-my-pi/pi-coding-agent/topics/board";
import { createTopicsHost, type TopicsHost } from "@oh-my-pi/pi-coding-agent/topics/host";
import { resetTopicsRegistryForTests, topicsRegistry } from "@oh-my-pi/pi-coding-agent/topics/workers";
import { VibeSessionRegistry } from "@oh-my-pi/pi-coding-agent/vibe/runtime";
import type { IrcMessage } from "@oh-my-pi/pi-coding-agent/irc/irc-types";
import type { AgentProgress, SingleResult } from "@oh-my-pi/pi-coding-agent/task/task-details";
import { TempDir } from "@oh-my-pi/pi-utils";

const OWNER = "Main";

type FollowUpOptions = Parameters<typeof executorModule.runSubagentFollowUpTurn>[0];

/** A follow-up turn held in flight until released with its output. */
interface HeldTurn {
	options: FollowUpOptions;
	release(output?: string): void;
}

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

describe("topics registry", () => {
	let tempDir: TempDir;
	let manager: AsyncJobManager;
	let sessionManager: SessionManager;
	let delivered: string[];
	/** Executor options of every first turn, in spawn order. */
	let spawnOptions: ExecutorOptions[];
	let hosts: TopicsHost[];

	function parentFor(sm: SessionManager): ToolSession {
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
	 * Run a worker's first turn to completion the way the executor leaves it:
	 * a child transcript with a session_init entry, and an idle agent ref.
	 */
	async function spawnSettled(
		registry: VibeSessionRegistry,
		parent: ToolSession,
		args: { name: string; title?: string },
	): Promise<string> {
		vi.spyOn(executorModule, "runSubprocess").mockImplementation(async options => {
			spawnOptions.push(options);
			const childFile = path.join(
				path.dirname(sessionManager.getSessionFile()!),
				path.basename(sessionManager.getSessionFile()!, ".jsonl"),
				`${options.id}.jsonl`,
			);
			fs.mkdirSync(path.dirname(childFile), { recursive: true });
			const header = {
				type: "session",
				version: 3,
				id: `${options.id}-session`,
				timestamp: new Date().toISOString(),
				cwd: tempDir.path(),
			};
			const init = {
				type: "session_init",
				id: "init1",
				parentId: null,
				timestamp: new Date().toISOString(),
				systemPrompt: "s",
				task: options.task,
				tools: [],
			};
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
		const spawned = await registry.spawn(parent, { cli: "good", prompt: "brief", ...args });
		await manager.getJob(spawned.jobId)?.promise;
		return spawned.id;
	}

	/** Hold every follow-up turn in flight; `turn(n)` resolves once the n-th (0-based) has started. */
	function holdFollowUps(): (index: number) => Promise<HeldTurn> {
		const slots: PromiseWithResolvers<HeldTurn>[] = [];
		const slot = (index: number) => (slots[index] ??= Promise.withResolvers<HeldTurn>());
		let calls = 0;
		vi.spyOn(executorModule, "runSubagentFollowUpTurn").mockImplementation(options => {
			const done = Promise.withResolvers<SingleResult>();
			const base = resultFor({
				...options,
				cwd: tempDir.path(),
				task: options.message,
				index: 0,
			} as ExecutorOptions);
			slot(calls++).resolve({
				options,
				release: (output = base.output) => done.resolve({ ...base, output }),
			});
			return done.promise;
		});
		return index => slot(index).promise;
	}

	/** Wait until turn `n` of worker `id` has settled. */
	async function settled(id: string, n: number): Promise<void> {
		await manager.getJob(`${id}-t${n}`)?.promise.catch(() => {});
	}

	function hostFor(parent: ToolSession): TopicsHost {
		const host = createTopicsHost(parent);
		hosts.push(host);
		return host;
	}

	/** Register the Inbox (main agent) as an IRC recipient; returns what it receives. */
	function registerInbox(): IrcMessage[] {
		const received: IrcMessage[] = [];
		AgentRegistry.global().register({
			id: OWNER,
			displayName: OWNER,
			kind: "main",
			session: {
				deliverIrcMessage: async (message: IrcMessage) => {
					received.push(message);
					return "injected" as const;
				},
			} as never,
			sessionFile: null,
			status: "idle",
		});
		return received;
	}

	/** `from` writes `body` to `agent://<to>`, as its `write` tool would. */
	async function ask(from: string, to: string, body: string): Promise<{ text: string; isError: boolean }> {
		const session = {
			getAgentId: () => from,
			agentRegistry: AgentRegistry.global(),
			settings: Settings.isolated(),
			taskDepth: from === OWNER ? 0 : 1,
			getSessionFile: () => null,
		} as unknown as ToolSession;
		const result = await InternalUrlRouter.instance().write(`agent://${to}`, body, { session });
		const first = result?.content[0];
		return { text: first?.type === "text" ? first.text : "", isError: result?.isError === true };
	}

	beforeEach(async () => {
		tempDir = TempDir.createSync("@omp-topics-");
		delivered = [];
		spawnOptions = [];
		hosts = [];
		manager = new AsyncJobManager({});
		manager.registerDeliverySink(OWNER, async (_jobId, text) => {
			delivered.push(text);
		});
		sessionManager = SessionManager.create(tempDir.path(), tempDir.path());
		sessionManager.appendMessage({ role: "user", content: "hello", timestamp: Date.now() });
		await sessionManager.ensureOnDisk();
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

	it("does not deliver a topic's turn result into the parent, while a vibe worker's still is", async () => {
		const parent = parentFor(sessionManager);
		await spawnSettled(topicsRegistry(), parent, { name: "Diokan", title: "Diokan term sheet" });
		await spawnSettled(VibeSessionRegistry.global(), parent, { name: "Worker" });
		await manager.drainDeliveries({ timeoutMs: 5_000 });

		expect(delivered.some(text => text.includes("answer from Worker"))).toBe(true);
		expect(delivered.some(text => text.includes("answer from Diokan"))).toBe(false);
	});

	it("starts topic workers at the owner's spawn depth, while vibe workers count one level down", async () => {
		const parent = parentFor(sessionManager);
		await spawnSettled(topicsRegistry(), parent, { name: "Diokan", title: "Diokan" });
		await spawnSettled(VibeSessionRegistry.global(), parent, { name: "Worker" });
		const [topicOptions, vibeOptions] = spawnOptions;

		// spawnDepth 0 = same nesting allowance as the main session; absent = executor's parent + 1.
		expect(topicOptions).toMatchObject({ taskDepth: 0, spawnDepth: 0 });
		expect(vibeOptions?.spawnDepth).toBeUndefined();
	});

	it("persists title and close across a restart and reopens with the conversation", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const id = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		await topics.rename(parent, id, "Diokan signing");
		await topics.close(parent, id);
		await expect(topics.send(parent, { session: id, message: "more" })).rejects.toThrow("is closed");

		// Restart: fresh registries over the same parent file.
		const file = sessionManager.getSessionFile()!;
		VibeSessionRegistry.resetGlobalForTests();
		resetTopicsRegistryForTests();
		const parent2 = parentFor(await SessionManager.open(file));
		const topics2 = topicsRegistry();

		expect(await VibeSessionRegistry.global().rehydrate(parent2)).toBe(0);
		expect(await topics2.rehydrate(parent2)).toBe(1);
		expect(topics2.describe(parent2)).toMatchObject([{ id, title: "Diokan signing", state: "closed" }]);

		await topics2.reopen(parent2, id);
		expect(topics2.describe(parent2)[0]?.state).toBe("idle");
		const followUp = vi
			.spyOn(executorModule, "runSubagentFollowUpTurn")
			.mockImplementation(async options =>
				resultFor({ ...options, cwd: tempDir.path(), task: options.message, index: 0 } as ExecutorOptions),
			);
		const outcome = await topics2.send(parent2, { session: id, message: "status?" });
		expect(outcome.mode).toBe("turn");
		await manager.waitForAll();
		expect(topics2.describe(parent2)[0]).toMatchObject({ state: "idle", turns: 2 });
		expect(followUp.mock.calls[0]?.[0].message).toBe("status?");
	});

	it("gives each topic turn a board of the other topics, and vibe workers none", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const first = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		const second = await spawnSettled(topics, parent, { name: "Notes", title: "Release notes" });
		await spawnSettled(VibeSessionRegistry.global(), parent, { name: "Worker" });
		const [firstOptions, secondOptions, vibeOptions] = spawnOptions;

		expect(firstOptions?.topicBoard).toBeUndefined();
		expect(secondOptions?.topicBoard).toContain(`- Diokan term sheet (id: ${first}): idle`);
		expect(secondOptions?.topicBoard).not.toContain(`(id: ${second})`);
		expect(secondOptions?.topicBoard).toContain("You are a topic too; you are not listed.");
		expect(vibeOptions?.topicBoard).toBeUndefined();

		const followUp = vi
			.spyOn(executorModule, "runSubagentFollowUpTurn")
			.mockImplementation(async options =>
				resultFor({ ...options, cwd: tempDir.path(), task: options.message, index: 0 } as ExecutorOptions),
			);
		await topics.send(parent, { session: first, message: "status?" });
		await manager.waitForAll();
		const board = followUp.mock.calls[0]?.[0].topicBoard;
		expect(board).toContain(`- Release notes (id: ${second}): idle`);
		expect(board).not.toContain(`(id: ${first})`);
	});

	it("lists ids that history:// resolves to the topic's conversation", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const id = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		expect(formatTopicBoard(topics.describe(parent), "inbox")).toContain(`(id: ${id})`);

		const ref = AgentRegistry.global().get(id)!;
		const entry = {
			type: "message",
			id: "m1",
			parentId: "init1",
			timestamp: new Date().toISOString(),
			message: { role: "user", content: "draft v2 is ready", timestamp: 1 },
		};
		fs.appendFileSync(ref.sessionFile!, `${JSON.stringify(entry)}\n`);
		const history = await InternalUrlRouter.instance().resolve(`history://${id}`);
		expect(history.content).toContain("draft v2 is ready");
	});

	it("keeps the board through a working topic's progress and changes it when the turn settles", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const id = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		const turn = holdFollowUps();
		await topics.send(parent, { session: id, message: "draft clause 4" });
		const held = await turn(0);

		const board = () => formatTopicBoard(topics.describe(parent), "inbox");
		const working = topicBoardMessage(board(), [])!;
		expect(working.content).toContain(`(id: ${id}): working; turn started`);
		const history: AgentMessage[] = [working];
		for (const lastIntent of ["Reading the contract", "Drafting clause 4", "Checking definitions"]) {
			held.options.onProgress?.({
				recentTools: [],
				toolCount: 1,
				recentOutput: [],
				lastIntent,
			} as unknown as AgentProgress);
			expect(topicBoardMessage(board(), history)).toBeUndefined();
		}

		held.release("Clause 4 drafted");
		await settled(id, 2);
		const after = topicBoardMessage(board(), history);
		expect(after?.content).toContain(`(id: ${id}): idle; last activity`);
		// The gist is the owner's own line, not the agent's result text.
		expect(after?.content).toContain('owner asked "draft clause 4"');
		expect(after?.content).not.toContain("Clause 4 drafted");
	});

	it("runs an Inbox ask as a topic turn with the sibling board, labelled and counted, and answers the Inbox", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const diokan = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		const notes = await spawnSettled(topics, parent, { name: "Notes", title: "Release notes" });
		const inbox = registerInbox();
		hostFor(parent);
		const turn = holdFollowUps();

		expect(await ask(OWNER, diokan, "is clause 4 final?")).toMatchObject({ isError: false });
		const held = await turn(0);
		expect(held.options.message).toContain("Asked by Inbox (`Main`):");
		expect(held.options.message).toContain("is clause 4 final?");
		expect(held.options.topicBoard).toContain(`(id: ${notes})`);
		expect(held.options.topicBoard).not.toContain(`(id: ${diokan})`);
		expect(topics.describe(parent).find(topic => topic.id === diokan)).toMatchObject({ state: "running", turns: 2 });

		held.release("Clause 4 is final.");
		await settled(diokan, 2);
		expect(inbox).toMatchObject([{ from: diokan, to: OWNER, body: "Clause 4 is final.", wakeRelay: true }]);
	});

	it("keeps one message route per session when a second host is created, and one turn per ask", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const diokan = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		registerInbox();
		const first = hostFor(parent);
		hostFor(parent);
		expect(ircRouteCountForTests()).toBe(1);
		// The first host going away leaves the second one's route in place.
		first.dispose();
		expect(ircRouteCountForTests()).toBe(1);
		const turn = holdFollowUps();

		expect(await ask(OWNER, diokan, "is clause 4 final?")).toMatchObject({ isError: false });
		const held = await turn(0);
		expect(topics.describe(parent).find(topic => topic.id === diokan)).toMatchObject({ queued: 0, turns: 2 });
		held.release();
		await settled(diokan, 2);
		expect(executorModule.runSubagentFollowUpTurn).toHaveBeenCalledTimes(1);
	});

	it("queues an ask to a busy topic for a turn of its own, and a closed topic refuses asks", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const diokan = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		const notes = await spawnSettled(topics, parent, { name: "Notes", title: "Release notes" });
		registerInbox();
		hostFor(parent);
		const turn = holdFollowUps();

		await topics.send(parent, { session: diokan, message: "owner work" });
		const ownerTurn = await turn(0);
		const queued = await ask(OWNER, diokan, "status?");
		expect(queued).toMatchObject({ isError: false });
		expect(queued.text).toContain(`Queued for ${diokan}`);
		expect(topics.describe(parent).find(topic => topic.id === diokan)?.queued).toBe(1);

		ownerTurn.release();
		const askTurn = await turn(1);
		expect(askTurn.options.message).toContain("Asked by Inbox (`Main`):");
		expect(askTurn.options.message).toContain("status?");
		expect(askTurn.options.message).not.toContain("owner work");
		askTurn.release();
		await settled(diokan, 3);

		await topics.close(parent, notes);
		const refused = await ask(OWNER, notes, "hello?");
		expect(refused.isError).toBe(true);
		expect(refused.text).toContain("is closed");
	});

	it("lets a topic ask another only in a turn its owner started, and routes the answer back to it", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const trip = await spawnSettled(topics, parent, { name: "Trip", title: "Trip planning" });
		const diokan = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		const notes = await spawnSettled(topics, parent, { name: "Notes", title: "Release notes" });
		registerInbox();
		hostFor(parent);
		const turn = holdFollowUps();

		// In a turn its owner started, Trip may ask Diokan.
		await topics.send(parent, { session: trip, message: "check the term sheet with Diokan" });
		const tripTurn = await turn(0);
		expect(await ask(trip, diokan, "is clause 4 final?")).toMatchObject({ isError: false });
		const diokanTurn = await turn(1);
		expect(diokanTurn.options.message).toContain(`Asked by \`${trip}\`:`);

		// In the turn Trip's ask started, neither Diokan nor a helper it runs can ask another topic...
		const chained = await ask(diokan, notes, "can you check this?");
		expect(chained.isError).toBe(true);
		expect(chained.text).toContain("no chains");
		AgentRegistry.global().register({
			id: "DiokanHelper",
			displayName: "DiokanHelper",
			kind: "sub",
			parentId: diokan,
			session: null,
			sessionFile: null,
			status: "running",
		});
		expect((await ask("DiokanHelper", notes, "can you check this?")).text).toContain("no chains");
		// ...but Diokan can answer Trip, and its answer reaches Trip when Trip's turn ends.
		expect(await ask(diokan, trip, "which version?")).toMatchObject({ isError: false });
		diokanTurn.release("Clause 4 is final.");
		await settled(diokan, 2);
		tripTurn.release();
		const replyTurn = await turn(2);
		expect(replyTurn.options.id).toBe(trip);
		expect(replyTurn.options.message).toContain(`Reply from \`${diokan}\`:`);
		expect(replyTurn.options.message).toContain("which version?");
		expect(replyTurn.options.message).toContain("Clause 4 is final.");

		// The answer started Trip's turn, so Trip cannot ask on from it either: no loop.
		expect((await ask(trip, diokan, "and clause 5?")).text).toContain("no chains");
		replyTurn.release();
		await settled(trip, 3);
	});

	it("runs an owner message sent during an agent's turn as an owner turn of its own, which may ask", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const trip = await spawnSettled(topics, parent, { name: "Trip", title: "Trip planning" });
		const diokan = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		registerInbox();
		hostFor(parent);
		const turn = holdFollowUps();
		// Trip's session is mid-stream, so an owner message could be steered into the running turn.
		const steer = vi.fn(async () => {});
		const ref = AgentRegistry.global().get(trip)!;
		AgentRegistry.global().unregister(trip, ref);
		AgentRegistry.global().register({
			id: trip,
			displayName: trip,
			kind: "sub",
			parentId: OWNER,
			session: { isStreaming: true, steer } as never,
			sessionFile: ref.sessionFile,
			status: "idle",
		});

		await ask(OWNER, trip, "what's the plan?");
		const askTurn = await turn(0);
		const sent = await topics.send(parent, { session: trip, message: "check clause 4 with Diokan" });
		expect(sent.mode).toBe("queued");
		expect(steer).not.toHaveBeenCalled();

		askTurn.release();
		const ownerTurn = await turn(1);
		expect(ownerTurn.options.message).toBe("check clause 4 with Diokan");
		expect(await ask(trip, diokan, "is clause 4 final?")).toMatchObject({ isError: false });
		const diokanTurn = await turn(2);
		expect(diokanTurn.options.message).toContain(`Asked by \`${trip}\`:`);
		diokanTurn.release();
		ownerTurn.release();
		await settled(trip, 3);
	});

	it("never starts a turn with a progress note from a turn another agent's message started", async () => {
		const parent = parentFor(sessionManager);
		const topics = topicsRegistry();
		const diokan = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		const notes = await spawnSettled(topics, parent, { name: "Notes", title: "Release notes" });
		const inbox = registerInbox();
		hostFor(parent);
		const turn = holdFollowUps();

		await ask(OWNER, diokan, "is clause 4 final?");
		const askTurn = await turn(0);
		expect(await ask(diokan, `${OWNER}?progress`, "checking the redlines")).toMatchObject({ isError: false });
		expect(inbox).toMatchObject([{ from: diokan, progress: true, noWake: true, body: "checking the redlines" }]);
		expect(await ask(diokan, `${notes}?progress`, "fyi: clause 4 moves")).toMatchObject({ isError: false });
		expect(topics.describe(parent).find(topic => topic.id === notes)).toMatchObject({ state: "idle", turns: 1 });
		askTurn.release();
		await settled(diokan, 2);

		// From a turn its owner started, a progress note is ordinary delivery.
		await topics.send(parent, { session: diokan, message: "tell the Inbox how it goes" });
		const ownerTurn = await turn(1);
		await ask(diokan, `${OWNER}?progress`, "halfway");
		expect(inbox.at(-1)).toMatchObject({ body: "halfway", progress: true });
		expect(inbox.at(-1)?.noWake).toBeUndefined();
		ownerTurn.release();
		await settled(diokan, 3);
	});
});
