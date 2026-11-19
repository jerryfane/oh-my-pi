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
 *   `history://<id>` reads one, `agent://<id>` messages it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { InternalUrlRouter } from "@oh-my-pi/pi-coding-agent/internal-urls";
import { IrcBus } from "@oh-my-pi/pi-coding-agent/irc/bus";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { ExecutorOptions } from "@oh-my-pi/pi-coding-agent/task/executor";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { formatTopicBoard } from "@oh-my-pi/pi-coding-agent/topics/board";
import { VibeSessionRegistry } from "@oh-my-pi/pi-coding-agent/vibe/runtime";
import type { IrcMessage } from "@oh-my-pi/pi-tui/tools/irc";
import type { SingleResult } from "@oh-my-pi/pi-tui/tools/task";
import { TempDir } from "@oh-my-pi/pi-utils";

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

describe("topics registry", () => {
	let tempDir: TempDir;
	let manager: AsyncJobManager;
	let sessionManager: SessionManager;
	let delivered: string[];
	/** Executor options of every first turn, in spawn order. */
	let spawnOptions: ExecutorOptions[];

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

	beforeEach(async () => {
		tempDir = TempDir.createSync("@omp-topics-");
		delivered = [];
		spawnOptions = [];
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
		await manager.dispose({ timeoutMs: 100 });
		VibeSessionRegistry.resetGlobalForTests();
		AgentRegistry.resetGlobalForTests();
		IrcBus.resetGlobalForTests();
		InternalUrlRouter.resetForTests();
		tempDir.removeSync();
	});

	it("does not deliver a topic's turn result into the parent, while a vibe worker's still is", async () => {
		const parent = parentFor(sessionManager);
		await spawnSettled(VibeSessionRegistry.topics(), parent, { name: "Diokan", title: "Diokan term sheet" });
		await spawnSettled(VibeSessionRegistry.global(), parent, { name: "Worker" });
		await manager.drainDeliveries({ timeoutMs: 5_000 });

		expect(delivered.some(text => text.includes("answer from Worker"))).toBe(true);
		expect(delivered.some(text => text.includes("answer from Diokan"))).toBe(false);
	});

	it("starts topic workers at the owner's spawn depth, while vibe workers count one level down", async () => {
		const parent = parentFor(sessionManager);
		await spawnSettled(VibeSessionRegistry.topics(), parent, { name: "Diokan", title: "Diokan" });
		await spawnSettled(VibeSessionRegistry.global(), parent, { name: "Worker" });
		const [topicOptions, vibeOptions] = spawnOptions;

		// spawnDepth 0 = same nesting allowance as the main session; absent = executor's parent + 1.
		expect(topicOptions).toMatchObject({ taskDepth: 0, spawnDepth: 0 });
		expect(vibeOptions?.spawnDepth).toBeUndefined();
	});

	it("persists title and close across a restart and reopens with the conversation", async () => {
		const parent = parentFor(sessionManager);
		const topics = VibeSessionRegistry.topics();
		const id = await spawnSettled(topics, parent, { name: "Diokan", title: "Diokan term sheet" });
		await topics.rename(parent, id, "Diokan signing");
		await topics.close(parent, id);
		await expect(topics.send(parent, { session: id, message: "more" })).rejects.toThrow("is closed");

		// Restart: fresh registries over the same parent file.
		const file = sessionManager.getSessionFile()!;
		VibeSessionRegistry.resetGlobalForTests();
		const parent2 = parentFor(await SessionManager.open(file));
		const topics2 = VibeSessionRegistry.topics();

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
		const topics = VibeSessionRegistry.topics();
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

	it("lists ids that history:// and agent:// resolve to the topic", async () => {
		const parent = parentFor(sessionManager);
		const topics = VibeSessionRegistry.topics();
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

		// Live, the topic's session takes the message the way IRC delivers to any agent.
		const received: IrcMessage[] = [];
		AgentRegistry.global().unregister(id, ref);
		AgentRegistry.global().register({
			id,
			displayName: id,
			kind: "sub",
			parentId: OWNER,
			session: {
				deliverIrcMessage: async (message: IrcMessage) => {
					received.push(message);
					return "injected" as const;
				},
			} as never,
			sessionFile: ref.sessionFile,
			status: "idle",
		});
		const owner = { ...parent, agentRegistry: AgentRegistry.global() } as unknown as ToolSession;
		const sent = await InternalUrlRouter.instance().write(`agent://${id}`, "please add clause 5", { session: owner });
		expect(sent?.isError).toBeFalsy();
		expect(received).toMatchObject([{ from: OWNER, to: id, body: "please add clause 5" }]);
	});
});
