/**
 * Contract: a transcript-bound prompt's preamble (a topic's seed) counts toward
 * the request it joins (#18 review). The preamble joins the prompt only at
 * admission, after every pre-prompt step, but pre-prompt context maintenance
 * sizes the prompt with it: a seed too large for the model's window promotes to
 * the configured larger model before the first request, as an inline seed
 * would. Once attached, the prompt's cached token estimate is the fresh count
 * of what was sent, not the size of the bare task it was before.
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import { Agent, Tokenizer } from "@oh-my-pi/pi-agent-core";
import type { Context, Model } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { createInMemoryAuthStorage } from "../helpers/agent-session-setup";

const MARKER = "SEED-CONTEXT-MARKER";
const TASK = "Continue the topic here.";
/** One valid Inbox message, about 78k characters: far beyond a 4,000-token window. */
const SEED = `<inbox-history>\n<message role="user">\n${MARKER} ${"the quarterly numbers look fine ".repeat(2_400)}\n</message>\n</inbox-history>`;

const sessions: AgentSession[] = [];

afterEach(async () => {
	for (const session of sessions.splice(0)) await session.dispose();
	vi.restoreAllMocks();
});

interface Bench {
	session: AgentSession;
	/** Each provider request, by the model it went to. */
	requests: { model: string; text: string }[];
	small: Model;
}

/**
 * A session on a 4,000-token model. With `promote`, compaction and context promotion are on and the model names a
 * 200,000-token promotion target.
 */
function bench({ promote }: { promote: boolean }): Bench {
	const requests: Bench["requests"] = [];
	const respond = (id: string) => (context: Context) => {
		requests.push({ model: id, text: JSON.stringify(context.messages) });
		return { content: ["ok"] };
	};
	const small = createMockModel({ id: "small-model", contextWindow: 4_000, handler: respond("small-model") });
	const big = createMockModel({ id: "big-model", contextWindow: 200_000, handler: respond("big-model") });
	if (promote) Object.assign(small, { contextPromotionTarget: "mock/big-model" });
	const auth = createInMemoryAuthStorage();
	auth.keys.setRuntime("mock", "test-key");
	const modelRegistry = new ModelRegistry(auth);
	const available = modelRegistry.getAvailable.bind(modelRegistry);
	vi.spyOn(modelRegistry, "getAvailable").mockImplementation(kind => [small, big, ...available(kind)]);
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: { model: small, systemPrompt: ["Topic"], tools: [] },
		convertToLlm,
		streamFn: (model, context, options) => (model.id === big.id ? big : small).stream(model, context, options),
	});
	const session = new AgentSession({
		agent,
		sessionManager: SessionManager.inMemory(),
		settings: Settings.isolated({
			"compaction.enabled": promote,
			"contextPromotion.enabled": promote,
			"retry.enabled": false,
			"todo.enabled": false,
			"advisor.enabled": false,
		}),
		modelRegistry,
	});
	sessions.push(session);
	return { session, requests, small };
}

/** Send `TASK` bound to a live transcript, with `SEED` as its preamble. */
async function promptSeeded(session: AgentSession): Promise<void> {
	await session.prompt(TASK, {
		attribution: "agent",
		transcriptSignal: new AbortController().signal,
		transcriptPreamble: SEED,
	});
	await session.waitForIdle();
}

describe("a bound prompt's preamble in context sizing (#18 review)", () => {
	it("promotes to the larger model before the first request when the seed does not fit", async () => {
		const { session, requests } = await bench({ promote: true });
		await promptSeeded(session);

		expect(requests.map(request => request.model)).toEqual(["big-model"]);
		expect(requests[0]!.text.split(MARKER)).toHaveLength(2);
		expect(requests[0]!.text.indexOf(TASK)).toBeGreaterThan(requests[0]!.text.indexOf(MARKER));
	});

	it("keeps the sent prompt's token estimate coherent: the context counts the seed it carries", async () => {
		const { session, requests, small } = await bench({ promote: false });
		await promptSeeded(session);

		expect(requests.map(request => request.model)).toEqual(["small-model"]);
		const sent = session.agent.state.messages.find(message => message.role === "user");
		if (!sent) throw new Error("expected the sent prompt");
		const fresh = new Tokenizer(small).countMessage(structuredClone(sent));
		expect(fresh).toBeGreaterThan(10_000);
		const counted = session.getContextBreakdown({ contextWindow: 200_000 })?.messagesTokens ?? 0;
		expect(counted).toBeGreaterThanOrEqual(fresh);
	});
});
