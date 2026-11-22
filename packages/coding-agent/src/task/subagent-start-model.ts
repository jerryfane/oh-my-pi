/**
 * The model a subagent starts on, as the executor selects it: its model patterns, with an auth-aware fallback to
 * the parent's active model ({@link resolveModelOverrideWithAuthFallback}), pinned to the parent's provider when
 * offline. The executor launches on it; topics check a seed against it before spawning.
 */
import { cfgOffline } from "../config/auxiliary-model-policy";
import type { ModelRegistry } from "../config/model-registry";
import { resolveModelOverride, resolveModelOverrideWithAuthFallback } from "../config/model-resolver";
import type { Settings } from "../config/settings";

export interface SubagentStartModelOptions {
	/** The agent's model patterns (its override, else its definition's). */
	modelPatterns: string[];
	parentActiveModelPattern?: string;
	parentActiveProvider?: string;
	/** Exact selectors only (per-call model grants). */
	exactOnly: boolean;
	modelRegistry: ModelRegistry;
	settings: Settings;
	/** The subagent's id: session-sticky credentials resolve for it. */
	id?: string;
}

/** The provider offline mode pins a subagent to (`null`: none resolves), or undefined when online. */
function offlineProvider(options: SubagentStartModelOptions): string | null | undefined {
	if (!cfgOffline.get(options.settings)) return undefined;
	const { parentActiveProvider, parentActiveModelPattern, modelRegistry, settings } = options;
	return (
		parentActiveProvider ??
		(parentActiveModelPattern
			? resolveModelOverride([parentActiveModelPattern], modelRegistry, settings).model?.provider
			: undefined) ??
		null
	);
}

/** The subagent's start model resolution, and the provider it is pinned to offline. */
export async function resolveSubagentStartModel(options: SubagentStartModelOptions) {
	const allowedProvider = offlineProvider(options);
	const resolution = await resolveModelOverrideWithAuthFallback(
		options.modelPatterns,
		options.parentActiveModelPattern,
		options.modelRegistry,
		options.settings,
		options.id,
		{ exactOnly: options.exactOnly, allowedProvider },
	);
	return { allowedProvider, resolution };
}
