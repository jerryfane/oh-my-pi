import * as path from "node:path";
import type { Message } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { runRpcMode } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-mode";
import { createAgentSession } from "@oh-my-pi/pi-coding-agent/sdk";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { cfgAsyncEnabled } from "@oh-my-pi/pi-coding-agent/tools/settings";

// Real SDK session with topics enabled and RPC dispatch. The model answers "pong", except an
// Organise turn, which it never finishes: the turn stays running for the rest of the process.
const MODEL = getBundledModel("anthropic", "claude-sonnet-4-5");
if (!MODEL) throw new Error("bundled test model missing");

const cwd = process.cwd();
const authStorage = await AuthStorage.create(path.join(cwd, "auth.db"));
authStorage.keys.setRuntime("anthropic", "test-key");
const modelRegistry = new ModelRegistry(authStorage, path.join(cwd, "models.yml"));
const settings = Settings.isolated({ "topics.enabled": true });
cfgAsyncEnabled.set(settings, false);

const { session } = await createAgentSession({
	cwd,
	agentDir: cwd,
	sessionManager: SessionManager.inMemory(cwd),
	authStorage,
	modelRegistry,
	settings,
	model: MODEL,
	disableExtensionDiscovery: true,
	skills: [],
	contextFiles: [],
	workspaceTree: { rootPath: cwd, rendered: "", truncated: false, totalLines: 0, agentsMdFiles: [] },
	promptTemplates: [],
	slashCommands: [],
	enableMCP: false,
	enableLsp: false,
});

function textOf(message: Message | undefined): string {
	const content = message?.content;
	if (typeof content === "string") return content;
	return (content ?? []).map(block => ("text" in block ? block.text : "")).join("");
}

const mock = createMockModel({
	handler: context =>
		textOf(context.messages.at(-1)).includes("call `topic_suggest`")
			? new Promise(() => {})
			: { content: ["pong"] },
});
session.agent.streamFn = (model, context, options) => mock.stream(model, context, options);
await runRpcMode(session);
