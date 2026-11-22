/**
 * Contract: a seeded topic's child launch is bound to the Inbox transcript its
 * seed was read from (#13 review). The real executor creates the child (the SDK
 * session, a mock provider); when `/tree` moves the Inbox off the seed's source
 * message while that child is still being created, the launch is cancelled:
 * no provider request carries the abandoned excerpt, the child's session file
 * holds none of it (no `session_init`, no prompt row), nothing links the topic
 * in the Inbox, and an accepted suggestion stays open.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { Agent } from "@oh-my-pi/pi-agent-core";
import { unregisterCustomApis } from "@oh-my-pi/pi-ai/api-registry";
import { createMockModel, registerMockApi } from "@oh-my-pi/pi-ai/providers/mock";
import { closeModelCache } from "@oh-my-pi/pi-catalog/model-cache";
import { AsyncJobManager } from "@oh-my-pi/pi-coding-agent/async/job-manager";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentLifecycleManager } from "@oh-my-pi/pi-coding-agent/registry/agent-lifecycle";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import * as sdkModule from "@oh-my-pi/pi-coding-agent/sdk";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AgentStorage } from "@oh-my-pi/pi-coding-agent/session/agent-storage";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { ExecutorOptions } from "@oh-my-pi/pi-coding-agent/task/executor";
import * as executorModule from "@oh-my-pi/pi-coding-agent/task/executor";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { createTopicsHost, type TopicsHost } from "@oh-my-pi/pi-coding-agent/topics/host";
import { sessionInbox } from "@oh-my-pi/pi-coding-agent/topics/inbox";
import { TOPIC_CONTINUED_TYPE, TOPIC_SUGGESTION_STATE_TYPE } from "@oh-my-pi/pi-coding-agent/topics/suggestions";
import { resetTopicsRegistryForTests } from "@oh-my-pi/pi-coding-agent/topics/workers";
import { __resetDirsFromEnvForTests, removeWithRetries, setAgentDir } from "@oh-my-pi/pi-utils";
import { createAssistantMessage, createInMemoryAuthStorage } from "../helpers/agent-session-setup";

const EXCERPT = "ABANDONED-EXCERPT about the Diokan deal";
const MOCK_API_SOURCE = "test/topics-seeded-launch";
const ENV_KEYS = ["HOME", "PI_CODING_AGENT_DIR", "OMP_PROFILE", "PI_PROFILE"] as const;

let savedEnv: Record<string, string | undefined> = {};
let root: string;
const sessions: AgentSession[] = [];
const managers: AsyncJobManager[] = [];

function restoreEnvValue(key: string, value: string | undefined): void {
	if (value === undefined) {
		delete process.env[key];
		delete Bun.env[key];
		return;
	}
	process.env[key] = value;
	Bun.env[key] = value;
}

beforeEach(async () => {
	savedEnv = Object.fromEntries(ENV_KEYS.map(key => [key, process.env[key]]));
	root = fs.mkdtempSync(path.join(os.tmpdir(), "omp-topics-seeded-launch-"));
	const home = path.join(root, "home");
	fs.mkdirSync(path.join(home, "work"), { recursive: true });
	restoreEnvValue("HOME", home);
	vi.spyOn(os, "homedir").mockReturnValue(home);
	setAgentDir(path.join(home, ".omp", "agent"));
	AgentRegistry.resetGlobalForTests();
	AgentLifecycleManager.resetGlobalForTests();
	registerMockApi(MOCK_API_SOURCE);
});

afterEach(async () => {
	for (const session of sessions.splice(0)) await session.dispose();
	for (const manager of managers.splice(0)) await manager.dispose({ timeoutMs: 100 });
	resetTopicsRegistryForTests();
	await AgentLifecycleManager.global().dispose();
	AgentLifecycleManager.resetGlobalForTests();
	AgentRegistry.resetGlobalForTests();
	unregisterCustomApis(MOCK_API_SOURCE);
	vi.restoreAllMocks();
	for (const key of ENV_KEYS) restoreEnvValue(key, savedEnv[key]);
	__resetDirsFromEnvForTests();
	AgentStorage.close();
	closeModelCache();
	await removeWithRetries(root);
});

interface Launch {
	inbox: AgentSession;
	manager: SessionManager;
	host: TopicsHost;
	jobs: AsyncJobManager;
	/** Every provider request of the child, as the text it was sent. */
	requests: string[];
	/** The id of the first answer (before the excerpt) and of the excerpt message. */
	firstAnswer: string;
	excerpt: string;
	/** Resolves once the real SDK child creation started; it waits for `release`. */
	creating: Promise<void>;
	release(): void;
	/** The options the real executor was called with (its artifacts dir holds the child's session file). */
	launched: ExecutorOptions[];
}

/** An Inbox whose branch holds `EXCERPT`, a topics host on it, and a gated real child launch. */
async function openLaunch(): Promise<Launch> {
	const cwd = path.join(root, "home", "work");
	const authStorage = createInMemoryAuthStorage();
	authStorage.keys.setRuntime("mock", "test-key");
	const modelRegistry = new ModelRegistry(authStorage);
	const requests: string[] = [];
	const mock = createMockModel({
		handler: context => {
			requests.push(JSON.stringify(context.messages));
			return { content: [{ type: "toolCall", name: "yield", arguments: { type: "result", data: "done" } }] };
		},
	});
	const catalogAvailable = modelRegistry.getAvailable.bind(modelRegistry);
	vi.spyOn(modelRegistry, "getAvailable").mockImplementation(kind => [mock, ...catalogAvailable(kind)]);
	const settings = Settings.isolated({
		"task.agentIdleTtlMs": 0,
		"async.enabled": false,
		"compaction.enabled": false,
		"retry.enabled": false,
		"todo.enabled": false,
		"todo.reminders": false,
		"advisor.enabled": false,
		modelRoles: { default: "mock/mock-model" },
	});

	const manager = SessionManager.inMemory(cwd);
	manager.appendMessage({ role: "user", content: "first question", timestamp: 1 });
	const firstAnswer = manager.appendMessage(createAssistantMessage("first answer"));
	const excerpt = manager.appendMessage({ role: "user", content: EXCERPT, timestamp: 2 });
	manager.appendMessage(createAssistantMessage("noted"));

	const jobs = new AsyncJobManager({});
	managers.push(jobs);
	const toolSession = {
		cwd,
		settings,
		authStorage,
		modelRegistry,
		asyncJobManager: jobs,
		sessionManager: manager,
		getAgentId: () => "Main",
		getSessionId: () => manager.getSessionId(),
		getSessionFile: () => null,
		getArtifactsDir: () => null,
		taskDepth: 0,
		enableLsp: false,
	} as unknown as ToolSession;
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: {
			model: mock,
			systemPrompt: ["Inbox"],
			tools: [],
			messages: manager.buildSessionContext().messages,
		},
		convertToLlm,
		streamFn: (model, context, options) => mock.stream(model, context, options),
	});
	const inbox: AgentSession = new AgentSession({
		agent,
		sessionManager: manager,
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry,
		createTopicsHost: () =>
			createTopicsHost(toolSession, { inbox: sessionInbox(inbox), summarize: async () => "summary" }),
	});
	sessions.push(inbox);
	const host = await inbox.getTopicsHost();
	if (!host) throw new Error("Expected a topics host");

	const launched: ExecutorOptions[] = [];
	const runSubprocess = executorModule.runSubprocess;
	vi.spyOn(executorModule, "runSubprocess").mockImplementation(options => {
		launched.push(options);
		return runSubprocess({ ...options, enableMCP: false, enableIrc: false });
	});
	const creating = Promise.withResolvers<void>();
	const gate = Promise.withResolvers<void>();
	const createAgentSession = sdkModule.createAgentSession;
	vi.spyOn(sdkModule, "createAgentSession").mockImplementation(async options => {
		creating.resolve();
		await gate.promise;
		return createAgentSession(options);
	});
	return {
		inbox,
		manager,
		host,
		jobs,
		requests,
		firstAnswer,
		excerpt,
		creating: creating.promise,
		release: () => gate.resolve(),
		launched,
	};
}

/** Everything the child's session file (if it was written) holds. */
function childFile(launch: Launch, topicId: string): string {
	const dir = launch.launched[0]?.artifactsDir;
	if (!dir) return "";
	const file = path.join(dir, `${topicId}.jsonl`);
	return fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
}

function rows(manager: SessionManager, customType: string): unknown[] {
	return manager.getEntries().filter(entry => "customType" in entry && entry.customType === customType);
}

async function settled(launch: Launch, topicId: string): Promise<void> {
	await launch.jobs.getJob(`${topicId}-t1`)?.promise.catch(() => {});
	for (let n = 0; n < 5; n++) await Bun.sleep(1);
}

describe("seeded child launch bound to the Inbox transcript (#13 review)", () => {
	it("Accept: /tree during the child's creation cancels the launch; nothing is sent, saved or linked", async () => {
		const launch = await openLaunch();
		const suggestion = launch.host.suggest({ title: "Diokan", reason: "r", messageIds: [launch.excerpt] });
		await Bun.sleep(1);
		const { topicId } = await launch.host.acceptSuggestion({ suggestionId: suggestion.id, by: "owner" });
		await launch.creating;
		const leaf = launch.manager.getLeafId();
		expect((await launch.inbox.navigateTree(launch.firstAnswer)).cancelled).toBe(false);
		expect(launch.manager.getBranch().some(entry => entry.id === launch.excerpt)).toBe(false);
		launch.release();
		await settled(launch, topicId);

		expect(launch.launched).toHaveLength(1);
		expect(launch.requests.filter(request => request.includes(EXCERPT))).toEqual([]);
		expect(childFile(launch, topicId)).not.toContain(EXCERPT);
		expect(rows(launch.manager, TOPIC_SUGGESTION_STATE_TYPE)).toEqual([]);
		expect(rows(launch.manager, TOPIC_CONTINUED_TYPE)).toEqual([]);
		// Back on the branch that holds its message, the suggestion is offered again.
		if (!leaf) throw new Error("Expected a leaf");
		await launch.inbox.navigateTree(leaf);
		expect(launch.host.suggestions().map(open => open.id)).toEqual([suggestion.id]);
	});

	it("Accept without a replacement: the child admits its seeded prompt, then the topic is linked", async () => {
		const launch = await openLaunch();
		const suggestion = launch.host.suggest({ title: "Diokan", reason: "r", messageIds: [launch.excerpt] });
		await Bun.sleep(1);
		const { topicId } = await launch.host.acceptSuggestion({ suggestionId: suggestion.id, by: "owner" });
		launch.release();
		await settled(launch, topicId);

		expect(launch.requests.some(request => request.includes(EXCERPT))).toBe(true);
		expect(childFile(launch, topicId)).toContain(EXCERPT);
		expect(rows(launch.manager, TOPIC_SUGGESTION_STATE_TYPE)).toHaveLength(1);
		expect(rows(launch.manager, TOPIC_CONTINUED_TYPE)).toHaveLength(1);
	});

	it("Create: /tree during the child's creation cancels the launch; nothing is sent, saved or linked", async () => {
		const launch = await openLaunch();
		const { id } = await launch.host.create({ title: "Diokan", message: "go", seedMessageIds: [launch.excerpt] });
		await launch.creating;
		expect((await launch.inbox.navigateTree(launch.firstAnswer)).cancelled).toBe(false);
		launch.release();
		await settled(launch, id);

		expect(launch.launched).toHaveLength(1);
		expect(launch.requests.filter(request => request.includes(EXCERPT))).toEqual([]);
		expect(childFile(launch, id)).not.toContain(EXCERPT);
		expect(rows(launch.manager, TOPIC_CONTINUED_TYPE)).toEqual([]);
	});
});
