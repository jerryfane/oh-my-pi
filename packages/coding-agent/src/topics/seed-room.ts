/**
 * Whether a seeded topic can start at all, decided when it is accepted or created, before anything is spawned. It
 * refuses only what is proven impossible: the seed's floor ({@link topicSeedFloor}), with nothing else of the
 * request, is larger than the window of every model the topic's child can use: the one its launch selects
 * ({@link VibeSessionRegistry.workerStartModel}) and that model's context-promotion target. Everything else is
 * left to the child, which fits the seed to its first request and reports whether that request started.
 */
import { Tokenizer } from "@oh-my-pi/pi-agent-core";
import type { Model } from "@oh-my-pi/pi-ai";
import { promotionTarget } from "../session/inbox-epoch";
import type { ToolSession } from "../tools";
import { TopicsError } from "./suggestions";

/** The models a child starting on `start` can send its first request to: `start`, and its promotion target. */
function childModels(session: ToolSession, start: Model): Model[] {
	const registry = session.modelRegistry;
	if (!registry) return [start];
	const fitted = registry.fitContextWindow(start, session.settings);
	const target = promotionTarget(fitted, session.settings, registry);
	return target ? [fitted, target] : [fitted];
}

/**
 * Refuse a seeded topic (`seed_too_large`) whose seed floor (`floor`: its smallest forms) is larger than the
 * window of every model a child starting on `start` can use. An unknown model or window refuses nothing.
 */
export function requireSeedRoom(session: ToolSession, start: Model | undefined, floor: readonly string[]): void {
	if (!start) return;
	let closest: { tokens: number; limit: number } | undefined;
	for (const model of childModels(session, start)) {
		const limit = model.contextWindow ?? 0;
		if (limit <= 0) return;
		const tokenizer = new Tokenizer(model);
		const tokens = Math.min(...floor.map(form => tokenizer.countTokens(form)));
		if (tokens <= limit) return;
		if (!closest || limit - tokens > closest.limit - closest.tokens) closest = { tokens, limit };
	}
	if (!closest) return;
	throw new TopicsError(
		"seed_too_large",
		`The copied history is too large for the available models, even shortened: ${closest.tokens} tokens, limit ${closest.limit}.`,
	);
}
