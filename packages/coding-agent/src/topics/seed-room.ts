/**
 * Whether a seeded topic can start at all, decided when it is accepted or created, before anything is saved or
 * spawned: its task plus the shortest seed ({@link shortestTopicSeed}) must fit the first request of a model the
 * topic's child can use (the one it starts on, or its context-promotion target), that model's window less the
 * output reserve and the request's system prompt and tools. Those are the child's own once it runs; the Inbox's
 * stand in for them here. Running, the child sizes its first request exactly and shortens the seed to fit it
 * ({@link BoundPrompt}); only a seed that cannot fit even there refuses its first turn.
 */
import { Tokenizer } from "@oh-my-pi/pi-agent-core";
import type { Model } from "@oh-my-pi/pi-ai";
import { resolveModelOverride } from "../config/model-resolver";
import { promotionTarget, promptLimit } from "../session/inbox-epoch";
import type { ToolSession } from "../tools";
import { TopicsError } from "./suggestions";

/** The models a child starting on `patterns` can send its first request to: its own, and its promotion target. */
function childModels(session: ToolSession, patterns: string[]): Model[] {
	const registry = session.modelRegistry;
	if (!registry) return [];
	const settings = session.settings;
	const parent = session.getActiveModelString?.();
	const start =
		resolveModelOverride(patterns, registry, settings).model ??
		(parent ? resolveModelOverride([parent], registry, settings).model : undefined);
	if (!start) return [];
	const target = promotionTarget(start, settings, registry);
	return target ? [registry.fitContextWindow(start, settings), target] : [registry.fitContextWindow(start, settings)];
}

/**
 * Refuse a seeded topic (`seed_too_large`) whose first request (`overheadTokens` of system prompt and tools, the
 * `task`, the shortest `seed`) fits no model a child starting on `patterns` can use. Unknown models or windows
 * refuse nothing: the child still checks its request.
 */
export function requireSeedRoom(
	session: ToolSession,
	patterns: string[],
	request: { task: string; seed: string; overheadTokens: number },
): void {
	let closest: { tokens: number; limit: number } | undefined;
	for (const model of childModels(session, patterns)) {
		const limit = promptLimit(model, session.settings);
		if (limit === undefined) return;
		const tokenizer = new Tokenizer(model);
		const tokens = request.overheadTokens + tokenizer.countTokens(request.task) + tokenizer.countTokens(request.seed);
		if (tokens <= limit) return;
		if (!closest || limit - tokens > closest.limit - closest.tokens) closest = { tokens, limit };
	}
	if (!closest) return;
	throw new TopicsError(
		"seed_too_large",
		`The copied history is too large for the available models, even shortened: ${closest.tokens} tokens, limit ${closest.limit}.`,
	);
}
