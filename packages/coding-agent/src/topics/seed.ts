/**
 * The seed of a topic started from Inbox messages (an accepted suggestion, or
 * `topics_new` with `seedMessageIds`): the newest {@link SEED_VERBATIM_MESSAGES}
 * chosen messages copied verbatim (user and assistant text only; tool calls and
 * results never), the older ones condensed into one summary by one small-model
 * call, and the suggestion's reason. A failed summary never stops the topic: the
 * older messages are cut and a line says how many.
 */
import { completeSimple } from "@oh-my-pi/pi-ai";
import { logger, prompt } from "@oh-my-pi/pi-utils";
import { cfgOffline } from "../config/auxiliary-model-policy";
import { resolveRoleChain } from "../config/model-resolver";
import seedTemplate from "../prompts/system/topic-seed.md" with { type: "text" };
import seedSummaryPrompt from "../prompts/system/topic-seed-summary.md" with { type: "text" };
import { escapeHarnessTags } from "../session/harness-tags";
import { journalCompletion } from "../session/model-usage-journal";
import type { SessionEntry } from "../session/session-entries";
import type { ToolSession } from "../tools";
import { StaleTranscriptError, throwIfStale } from "../session/inbox-epoch";
import { messageText } from "./suggestions";

/** Chosen messages copied verbatim; older ones are summarised. */
export const SEED_VERBATIM_MESSAGES = 30;
/** Longest transcript sent to the summary model; the oldest text is cut beyond it. */
const SUMMARY_INPUT_MAX_CHARS = 120_000;
const SUMMARY_TIMEOUT_MS = 60_000;

/** The Inbox transcript a seed comes from: its session id, and its epoch's signal (aborted once it is replaced). */
export interface SeedOrigin {
	sessionId: string;
	signal: AbortSignal;
}

/**
 * Condense `transcript` (older messages of the Inbox transcript `origin`) into one summary; undefined (or a
 * rejection) = failed. It works for `origin` only: an aborted origin cancels it ({@link StaleTranscriptError}).
 */
export type SeedSummarizer = (transcript: string, origin: SeedOrigin) => Promise<string | undefined>;

interface SeedMessage {
	role: "user" | "assistant";
	text: string;
}

/** `<` of a tag of the seed's own block (`inbox-history`, `message`), opening or closing. */
const SEED_TAG_START_RE = /<(?=\s*\/?\s*(?:inbox-history|message)(?![\w-]))/gi;

/**
 * Inbox text as it goes into the seed: no text can close or open the seed's block or a message of it (or a
 * harness block): the `<` of such a tag becomes `&lt;`, which the model still reads as the text it was.
 */
function seedText(text: string): string {
	return escapeHarnessTags(text).replace(SEED_TAG_START_RE, "&lt;");
}

function transcriptOf(messages: readonly SeedMessage[]): string {
	return messages
		.map(message => `<message role="${message.role}">\n${seedText(message.text)}\n</message>`)
		.join("\n\n");
}

/**
 * The seed text for the messages `messageIds` names in `entries` (file order;
 * ids that name no user or assistant text message are skipped).
 */
export async function buildTopicSeed(args: {
	entries: readonly SessionEntry[];
	messageIds: readonly string[];
	reason?: string;
	summarize: (transcript: string) => Promise<string | undefined>;
}): Promise<string> {
	const wanted = new Set(args.messageIds);
	const messages: SeedMessage[] = [];
	for (const entry of args.entries) {
		if (!wanted.has(entry.id)) continue;
		const message = messageText(entry);
		if (message) messages.push(message);
	}
	const older = messages.slice(0, Math.max(0, messages.length - SEED_VERBATIM_MESSAGES));
	const recent = messages.slice(older.length);
	let summary: string | undefined;
	if (older.length > 0) {
		try {
			summary = (await args.summarize(transcriptOf(older).slice(-SUMMARY_INPUT_MAX_CHARS)))?.trim() || undefined;
		} catch (error) {
			// A replaced transcript cancels the seed; it is not a failed summary to fall back from.
			if (error instanceof StaleTranscriptError) throw error;
			logger.warn("topics: seed summary failed", { error: error instanceof Error ? error.message : String(error) });
		}
	}
	const header = prompt
		.render(seedTemplate, {
			reason: args.reason?.trim() ? seedText(args.reason.trim()) : "",
			summary: summary ? seedText(summary) : "",
			older: older.length,
			unsummarised: older.length > 0 && !summary ? older.length : 0,
		})
		.trim();
	const transcript = transcriptOf(recent);
	return `<inbox-history>\n${header}${transcript ? `\n\n${transcript}` : ""}\n</inbox-history>`;
}

/**
 * The summarizer for `session`'s topics: one call to the `smol` role model,
 * the configured role or else its default priority chain (same provider only
 * when offline), cost-gated like other auxiliary calls and journaled in the
 * origin transcript only. No usable model counts as a failed summary.
 */
export function smolSeedSummarizer(session: ToolSession): SeedSummarizer {
	return async (transcript, origin) => {
		throwIfStale(origin.signal);
		const registry = session.modelRegistry;
		if (!registry) return undefined;
		const offline = cfgOffline.get(session.settings);
		const activeProvider = session.getActiveModel?.()?.provider;
		if (offline && !activeProvider) return undefined;
		const available = registry.getAvailable();
		const model = resolveRoleChain(
			"smol",
			session.settings,
			offline ? available.filter(candidate => candidate.provider === activeProvider) : available,
		)[0]?.model;
		if (!model) return undefined;
		const sessionId = origin.sessionId;
		const journal = journalCompletion(session.sessionManager, { purpose: "topic-seed-summary", role: "smol" });
		session.costGate?.admit();
		const message = await completeSimple(
			model,
			{
				systemPrompt: [prompt.render(seedSummaryPrompt)],
				messages: [{ role: "user", content: transcript, timestamp: Date.now() }],
			},
			{
				apiKey: registry.resolver(model, sessionId),
				sessionId,
				maxTokens: 1024,
				disableReasoning: true,
				signal: AbortSignal.any([origin.signal, AbortSignal.timeout(SUMMARY_TIMEOUT_MS)]),
			},
		);
		session.costGate?.record(message.usage.cost.total);
		// The origin was replaced while it ran: cancelled, and not journaled in the successor.
		throwIfStale(origin.signal);
		journal?.(message, transcript);
		if (message.stopReason === "error" || message.stopReason === "aborted") return undefined;
		return message.content
			.map(block => (block.type === "text" ? block.text : ""))
			.join("")
			.trim();
	};
}
