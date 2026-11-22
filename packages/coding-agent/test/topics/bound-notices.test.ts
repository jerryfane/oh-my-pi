/**
 * Contract: a transcript-bound prompt (a topic's seeded first prompt) is sized
 * with the notices its batch carries, and their delivery state is committed only
 * when the bound prompt admits that batch (#21 review). A batch refused at
 * admission (its seed does not fit even shortened) sends nothing, so the tool
 * roster change its notice announced is still pending: the next prompt carries
 * that notice.
 */
import { afterEach, describe, expect, it } from "bun:test";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import type { Message } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { PreambleTooLargeError, type TranscriptPreamble } from "@oh-my-pi/pi-coding-agent/session/inbox-epoch";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";

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

/** A session on a prefix-binding model (roster changes travel as a notice), recording every request's text. */
function bench(): { session: AgentSession; requests: string[] } {
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
	const model = { ...catalogued, contextWindow: 8_000, maxTokens: 2_000 };
	const read = tool("read");
	const bash = tool("bash");
	const mock = createMockModel({ responses: [{ content: ["ok"] }, { content: ["ok"] }] });
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
		]),
		builtInToolNames: ["read", "bash"],
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

describe("notices of a transcript-bound prompt (#21 review)", () => {
	const sessions: AgentSession[] = [];

	afterEach(async () => {
		for (const session of sessions.splice(0)) await session.dispose();
	});

	it("keeps a roster notice pending when its bound batch is refused at admission; the next prompt carries it", async () => {
		const { session, requests } = bench();
		sessions.push(session);
		await session.setActiveToolPresentation(["read"], []);
		await session.prompt("first");
		await session.waitForIdle();
		await session.setActiveToolPresentation(["read", "bash"], []);

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
});
