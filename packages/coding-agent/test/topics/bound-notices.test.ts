/**
 * Contract: a prompt's batch is built once, its notices included, and that one
 * batch is admitted (a transcript-bound prompt, a topic's seeded first prompt)
 * and sent; the deltas its notices announce are recorded as delivered only once
 * it is admitted (#21, #22, #23 review). A batch refused at admission (its seed
 * does not fit even shortened) sends nothing, so the tool roster change its
 * notice announced is still pending: the next prompt carries that notice. An
 * admitted one carries it once. A late delivery records exactly what its
 * notice announced when it was taken: a roster change or a mount since stays
 * pending, and so does a mount a later base rebuild lists.
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import type { Message } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import type { CustomTool } from "@oh-my-pi/pi-coding-agent/extensibility/custom-tools/types";
import type { ExtensionRunner } from "@oh-my-pi/pi-coding-agent/extensibility/extensions";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { PreambleTooLargeError, type TranscriptPreamble } from "@oh-my-pi/pi-coding-agent/session/inbox-epoch";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { SessionTools } from "@oh-my-pi/pi-coding-agent/session/session-tools";
import type { XdevState } from "@oh-my-pi/pi-coding-agent/tools/xdev";

const ROSTER_NOTICE = "Now available: bash.";

function tool(name: string): AgentTool {
	return {
		name,
		label: name,
		description: `${name} tool`,
		parameters: type({ value: "string" }),
		strict: true,
		async execute() {
			return { content: [{ type: "text", text: `${name} executed` }] };
		},
	};
}

/** A model that binds signed thinking to its prompt prefix: roster and mount changes travel as notices. */
function prefixBoundModel() {
	const catalogued = buildModel({
		id: "claude-fable-5-1",
		name: "Claude Fable 5.1",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "https://api.anthropic.com",
		reasoning: true,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8_000,
		maxTokens: 2_000,
	});
	// The catalog knows this model's real window; the bench needs a small one.
	return { ...catalogued, contextWindow: 8_000, maxTokens: 2_000 };
}

/** A session on a prefix-binding model, recording every request's text. */
function bench(): { session: AgentSession; requests: string[] } {
	const model = prefixBoundModel();
	const read = tool("read");
	const bash = tool("bash");
	const edit = tool("edit");
	const mock = createMockModel({ responses: [{ content: ["ok"] }, { content: ["ok"] }, { content: ["ok"] }] });
	const requests: string[] = [];
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: { model, systemPrompt: ["initial"], tools: [read], messages: [] },
		convertToLlm,
		streamFn: async (requestModel, context, options) => {
			requests.push(text(context.messages));
			return mock.stream(requestModel, context, options);
		},
	});
	const session = new AgentSession({
		agent,
		sessionManager: SessionManager.inMemory(),
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: { getApiKey: async () => "test-key" } as never,
		toolRegistry: new Map([
			[read.name, read],
			[bash.name, bash],
			[edit.name, edit],
		]),
		builtInToolNames: ["read", "bash", "edit"],
		rebuildSystemPrompt: async toolNames => ({ systemPrompt: [`tools:${toolNames.join(",")}`] }),
	});
	return { session, requests };
}

function text(messages: Message[]): string {
	return messages
		.flatMap(message =>
			typeof message.content === "string"
				? [message.content]
				: message.content.flatMap(part => (part.type === "text" ? [part.text] : [])),
		)
		.join("\n");
}

/** A preamble no form of which fits an 8,000-token window. */
const tooLarge: TranscriptPreamble = {
	text: Array.from({ length: 20_000 }, (_, n) => `entry ${n} of the copied history`).join("\n"),
	fit: (_maxTokens, count) => ({ text: tooLarge.text, tokens: count(tooLarge.text) }),
};

describe("notices of a prompt's batch (#21, #22, #23 review)", () => {
	const sessions: AgentSession[] = [];

	afterEach(async () => {
		for (const session of sessions.splice(0)) await session.dispose();
		vi.restoreAllMocks();
	});

	/** A session past its first request, with a roster change (bash added) pending. */
	async function withPendingRoster(): Promise<{ session: AgentSession; requests: string[] }> {
		const { session, requests } = bench();
		sessions.push(session);
		await session.setActiveToolPresentation(["read"], []);
		await session.prompt("first");
		await session.waitForIdle();
		await session.setActiveToolPresentation(["read", "bash"], []);
		return { session, requests };
	}

	it("keeps a roster notice pending when its bound batch is refused at admission; the next prompt carries it", async () => {
		const { session, requests } = await withPendingRoster();

		const refused = await session
			.prompt("seeded", {
				attribution: "agent",
				transcriptSignal: new AbortController().signal,
				transcriptPreamble: tooLarge,
			})
			.then(
				() => undefined,
				(error: unknown) => error,
			);
		expect(refused).toBeInstanceOf(PreambleTooLargeError);
		expect(requests).toHaveLength(1);

		await session.prompt("next");
		await session.waitForIdle();
		expect(requests).toHaveLength(2);
		expect(requests[1]).toContain(ROSTER_NOTICE);
	});

	it("delivers the roster notice once with an admitted bound batch; the next prompt has none", async () => {
		const { session, requests } = await withPendingRoster();
		const fits: TranscriptPreamble = {
			text: "history",
			fit: (_maxTokens, count) => ({ text: "history", tokens: count("history") }),
		};
		await session.prompt("seeded", {
			attribution: "agent",
			transcriptSignal: new AbortController().signal,
			transcriptPreamble: fits,
		});
		await session.waitForIdle();
		await session.prompt("next");
		await session.waitForIdle();

		expect(requests).toHaveLength(3);
		expect(requests[1]!.split(ROSTER_NOTICE)).toHaveLength(2);
		expect(requests[2]!.split(ROSTER_NOTICE)).toHaveLength(2);
	});

	it("a late delivery records only the roster delta its notice announced: a change since stays pending", async () => {
		const tools = captureTools();
		const { session, requests } = await withPendingRoster();
		const notice = tools().takePendingToolRosterNotice({ baseDelivered: false });
		expect(notice).toMatchObject({ details: { added: ["bash"], removed: [] } });
		// A newer change lands before the batch carrying the notice is delivered.
		await session.setActiveToolPresentation(["read", "bash", "edit"], []);
		tools().deliverNotices([notice!]);

		await session.prompt("next");
		await session.waitForIdle();
		expect(requests[1]).toContain("Now available: edit.");
		expect(requests[1]).not.toContain(ROSTER_NOTICE);
	});

	it("a late delivery records only the mount its notice announced: a mount since stays in the next notice (#23 review)", async () => {
		const tools = captureTools();
		const { session, requests } = await xdevBench();
		await session.refreshMCPTools([SEARCH]);
		const peeked = tools().peekPendingXdevMountNotice({ baseCatalogDelivered: false });
		const notice = tools().takePendingXdevMountNotice({
			baseCatalogDelivered: false,
			expectedContentKey: peeked?.contentKey ?? "",
		});
		expect(notice).toMatchObject({ details: { added: ["mcp__nucleus_search"] } });
		// A second mount and a base rebuild land before the batch carrying the notice is delivered.
		await session.refreshMCPTools([SEARCH, FETCH]);
		await session.refreshBaseSystemPrompt();
		tools().deliverNotices([notice!]);

		await session.prompt("next");
		await session.waitForIdle();
		expect(requests[1]).toContain("xd://mcp__nucleus_fetch");
		expect(requests[1]).not.toContain("xd://mcp__nucleus_search");
	});

	it("a stale mount preview records only the base-carried mounts it saw: a mount a later rebuild lists stays in the next notice (#23 review)", async () => {
		const tools = captureTools();
		// The next request's prompt is replaced (before_agent_start): its notice is all that announces a mount.
		const { session, requests } = await xdevBench({ prefixBound: false, replacePrompt: true });
		await session.refreshMCPTools([SEARCH]);
		// The preview no longer matches (maintenance changed it); the base this request carries is delivered.
		expect(tools().takePendingXdevMountNotice({ baseCatalogDelivered: true, expectedContentKey: "stale" })).toBe(
			undefined,
		);
		await session.refreshMCPTools([SEARCH, FETCH]);
		await session.refreshBaseSystemPrompt();

		await session.prompt("next");
		await session.waitForIdle();
		expect(requests[1]).toContain("xd://mcp__nucleus_fetch");
	});
});

/** The session tools of the next session built: taken from its first roster take. */
function captureTools(): () => SessionTools {
	let tools: SessionTools | undefined;
	const take = SessionTools.prototype.takePendingToolRosterNotice;
	vi.spyOn(SessionTools.prototype, "takePendingToolRosterNotice").mockImplementation(
		function (this: SessionTools, options) {
			tools = this;
			return take.call(this, options);
		},
	);
	return () => {
		if (!tools) throw new Error("expected the session's tools");
		return tools;
	};
}

function mcpTool(name: string): CustomTool {
	return { ...tool(`mcp__nucleus_${name}`), mcpServerName: "nucleus", mcpToolName: name } as unknown as CustomTool;
}

const SEARCH = mcpTool("search");
const FETCH = mcpTool("fetch");

/**
 * A session that mounts MCP tools as `xd://` devices (its rebuilt prompt lists the mounted catalog), past its first
 * request: prefix-bound (its prompt frozen, so mount changes travel as notices) unless `prefixBound` is false. With
 * `replacePrompt`, an extension replaces the system prompt of every request (no base catalog is delivered).
 */
async function xdevBench({ prefixBound = true, replacePrompt = false } = {}): Promise<{
	session: AgentSession;
	requests: string[];
}> {
	const bound = prefixBoundModel();
	const model = prefixBound
		? bound
		: { ...bound, thinking: bound.thinking && { ...bound.thinking, prefixBinding: false } };
	const read = tool("read");
	const write = tool("write");
	// Devices are read and written through `xd://`: the session has both tools.
	const xdev: XdevState = {
		tools: new Map([
			[read.name, read],
			[write.name, write],
		]),
		mountedNames: new Set(),
		builtInNames: new Set(["read", "write"]),
		isActive: () => true,
	};
	const mock = createMockModel({ responses: [{ content: ["ok"] }, { content: ["ok"] }] });
	const requests: string[] = [];
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: { model, systemPrompt: ["initial"], tools: [read, write], messages: [] },
		convertToLlm,
		streamFn: async (requestModel, context, options) => {
			requests.push(text(context.messages));
			return mock.stream(requestModel, context, options);
		},
	});
	const session = new AgentSession({
		agent,
		sessionManager: SessionManager.inMemory(),
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: { getApiKey: async () => "test-key" } as never,
		toolRegistry: xdev.tools,
		builtInToolNames: ["read", "write"],
		ensureWriteRegistered: async () => true,
		extensionRunner: replacePrompt
			? ({
					emitBeforeAgentStart: async () => ({ systemPrompt: ["extension replacement"] }),
					emit: async () => undefined,
				} as unknown as ExtensionRunner)
			: undefined,
		rebuildSystemPrompt: async (toolNames, _tools, state) => {
			const catalog = state ? [...state.mountedNames] : [];
			return {
				systemPrompt: [`tools:${toolNames.join(",")}\nxd:// catalog: ${catalog.join(",")}`],
				xdevCatalogNames: catalog,
			};
		},
		xdev,
	});
	sessionsToDispose.push(session);
	await session.prompt("first");
	await session.waitForIdle();
	return { session, requests };
}

const sessionsToDispose: AgentSession[] = [];
afterEach(async () => {
	for (const session of sessionsToDispose.splice(0)) await session.dispose();
});
