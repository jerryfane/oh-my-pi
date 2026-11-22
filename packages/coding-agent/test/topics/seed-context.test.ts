/**
 * Contract: a transcript-bound prompt's preamble (a topic's seed) counts toward
 * the request it joins (#18, #19 review). The preamble joins the prompt only at
 * admission, after every pre-prompt step, but pre-prompt context maintenance
 * sizes the prompt with it: a seed too large for the model's window promotes to
 * the configured larger model before the first request, as an inline seed
 * would. Once attached, the prompt's cached token estimate is the fresh count
 * of what was sent, not the size of the bare task it was before. A seed that
 * still does not fit the model the request goes to (its window less the output
 * reserve) is shortened at admission: the oldest messages are left out, then
 * the newest one's text is cut, each with a line that says so. When even the
 * shortest seed does not fit, the prompt is refused: nothing is sent or saved,
 * and the seed is told the size and the limit.
 */
import { afterEach, describe, expect, it, vi } from "bun:test";
import { Agent, Tokenizer } from "@oh-my-pi/pi-agent-core";
import { promptBudgetTokens } from "@oh-my-pi/pi-agent-core/compaction";
import type { Context, Model } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { PreambleTooLargeError } from "@oh-my-pi/pi-coding-agent/session/inbox-epoch";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { resolveModelCompactionSettings } from "@oh-my-pi/pi-coding-agent/session/model-compaction-threshold";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { buildTopicSeed, type TopicSeed } from "@oh-my-pi/pi-coding-agent/topics/seed";
import { createInMemoryAuthStorage } from "../helpers/agent-session-setup";

const MARKER = "SEED-CONTEXT-MARKER";
const OLDER = "OLDER-SEED-MESSAGE about the plan";
const TASK = "Continue the topic here.";
/** One valid Inbox message, about 78k characters: far beyond a 4,000-token window. */
const BIG = `${MARKER} ${"the quarterly numbers look fine ".repeat(2_400)}`;

const sessions: AgentSession[] = [];

afterEach(async () => {
	for (const session of sessions.splice(0)) await session.dispose();
	vi.restoreAllMocks();
});

/** A topic's seed of two Inbox messages: a short older one, then `BIG`. */
async function topicSeed(): Promise<TopicSeed> {
	const inbox = SessionManager.inMemory();
	const older = inbox.appendMessage({ role: "user", content: OLDER, timestamp: 1 });
	const big = inbox.appendMessage({ role: "user", content: BIG, timestamp: 2 });
	return buildTopicSeed({ entries: inbox.getEntries(), messageIds: [older, big], summarize: async () => undefined });
}

interface Bench {
	session: AgentSession;
	/** Each provider request, by the model it went to. */
	requests: { model: string; text: string }[];
	small: Model;
}

/**
 * A session on a `window`-token model, with context promotion on, and compaction unless `compaction` is false
 * (then no turn ends in a compaction of its own). With `target`, the model names a 200,000-token promotion target.
 */
function bench({
	window,
	target,
	compaction = true,
}: {
	window: number;
	target: boolean;
	compaction?: boolean;
}): Bench {
	const requests: Bench["requests"] = [];
	const respond = (id: string) => (context: Context) => {
		requests.push({ model: id, text: JSON.stringify(context.messages) });
		return { content: ["ok"] };
	};
	const small = createMockModel({ id: "small-model", contextWindow: window, handler: respond("small-model") });
	const big = createMockModel({ id: "big-model", contextWindow: 200_000, handler: respond("big-model") });
	if (target) Object.assign(small, { contextPromotionTarget: "mock/big-model" });
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
			"compaction.enabled": compaction,
			"contextPromotion.enabled": true,
			"retry.enabled": false,
			"todo.enabled": false,
			"advisor.enabled": false,
		}),
		modelRegistry,
	});
	sessions.push(session);
	return { session, requests, small };
}

/** Send `TASK` bound to a live transcript, with `seed` as its preamble. */
async function promptSeeded(session: AgentSession, seed: TopicSeed): Promise<void> {
	await session.prompt(TASK, {
		attribution: "agent",
		transcriptSignal: new AbortController().signal,
		transcriptPreamble: seed,
	});
	await session.waitForIdle();
}

describe("a bound prompt's preamble in context sizing (#18, #19 review)", () => {
	it("promotes to the larger model before the first request when the seed does not fit", async () => {
		const { session, requests } = bench({ window: 4_000, target: true });
		await promptSeeded(session, await topicSeed());

		expect(requests.map(request => request.model)).toEqual(["big-model"]);
		const [first] = requests;
		expect(first!.text.split(MARKER)).toHaveLength(2);
		expect(first!.text).toContain(OLDER);
		expect(first!.text).not.toContain("[truncated:");
		expect(first!.text.indexOf(TASK)).toBeGreaterThan(first!.text.indexOf(MARKER));
	});

	it("keeps the sent prompt's token estimate coherent: the context counts the seed it carries", async () => {
		const { session, requests, small } = bench({ window: 200_000, target: false });
		await promptSeeded(session, await topicSeed());

		expect(requests.map(request => request.model)).toEqual(["small-model"]);
		const sent = session.agent.state.messages.find(message => message.role === "user");
		if (!sent) throw new Error("expected the sent prompt");
		const fresh = new Tokenizer(small).countMessage(structuredClone(sent));
		expect(fresh).toBeGreaterThan(10_000);
		const counted = session.getContextBreakdown({ contextWindow: 200_000 })?.messagesTokens ?? 0;
		expect(counted).toBeGreaterThanOrEqual(fresh);
	});

	it("a 4,000-token model only: the seed is shortened so the first request fits, and says what it cut", async () => {
		const { session, requests, small } = bench({ window: 4_000, target: false, compaction: false });
		const seed = await topicSeed();
		await promptSeeded(session, seed);

		expect(requests.map(request => request.model)).toEqual(["small-model"]);
		const [first] = requests;
		// The older message is left out with a line that says so; the newest is cut from its end, with a marker.
		expect(first!.text).not.toContain(OLDER);
		expect(first!.text).toContain("1 earlier messages left out");
		expect(first!.text.split(MARKER)).toHaveLength(2);
		expect(first!.text).toMatch(/\[truncated: \d+ characters\]/);
		expect(first!.text.indexOf(TASK)).toBeGreaterThan(first!.text.indexOf(MARKER));
		const assistant = session.agent.state.messages.find(message => message.role === "assistant");
		if (!assistant) throw new Error("expected the answer");
		// The request: everything in context before the answer.
		const request =
			(session.getContextBreakdown({ contextWindow: 4_000 })?.usedTokens ?? Infinity) -
			session.agent.tokenizer.countMessage(assistant);
		const limit = promptBudgetTokens(4_000, resolveModelCompactionSettings(session.settings, small));
		expect(request).toBeLessThanOrEqual(limit);
		expect(request).toBeGreaterThan(limit / 2);
		expect(seed.refusal).toBeUndefined();
	});

	it("refuses the prompt when even the shortest seed does not fit: nothing is sent or saved", async () => {
		const { session, requests } = bench({ window: 60, target: false });
		const seed = await topicSeed();
		const refused = await promptSeeded(session, seed).then(
			() => undefined,
			(error: unknown) => error,
		);

		expect(refused).toBeInstanceOf(PreambleTooLargeError);
		expect(requests).toEqual([]);
		expect(session.agent.state.messages.filter(message => message.role === "user")).toEqual([]);
		expect(seed.refusal).toBeDefined();
		expect(seed.refusal!.tokens).toBeGreaterThan(seed.refusal!.limit);
		expect(seed.refusal).toEqual({
			tokens: (refused as PreambleTooLargeError).tokens,
			limit: (refused as PreambleTooLargeError).limit,
		});
	});
});
