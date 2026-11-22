/**
 * The seed of a topic started from Inbox messages (an accepted suggestion, or
 * `topics_new` with `seedMessageIds`): the newest {@link SEED_VERBATIM_MESSAGES}
 * chosen messages copied verbatim (user and assistant text only; tool calls and
 * results never), the older ones condensed into one summary by one small-model
 * call, and the suggestion's reason. A failed summary never stops the topic: the
 * older messages are cut and a line says how many. A seed too large for the
 * topic's first request is shortened to fit ({@link TopicSeed}).
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
import { StaleTranscriptError, throwIfStale, type TranscriptPreamble } from "../session/inbox-epoch";
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
	/** Characters cut from the end of the message's text so the seed fits its request. */
	cut?: number;
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
		.map(message => {
			const cut = message.cut ? `${message.text ? "\n" : ""}[truncated: ${message.cut} characters]` : "";
			return `<message role="${message.role}">\n${seedText(message.text)}${cut}\n</message>`;
		})
		.join("\n\n");
}

/** The chosen messages of a seed, file order: `older` ones are summarised, `recent` ones copied verbatim. */
interface SeedSelection {
	reason: string;
	older: SeedMessage[];
	recent: SeedMessage[];
}

function selectSeed(args: {
	entries: readonly SessionEntry[];
	messageIds: readonly string[];
	reason?: string;
}): SeedSelection {
	const wanted = new Set(args.messageIds);
	const messages: SeedMessage[] = [];
	for (const entry of args.entries) {
		if (!wanted.has(entry.id)) continue;
		const message = messageText(entry);
		if (message) messages.push(message);
	}
	const older = messages.slice(0, Math.max(0, messages.length - SEED_VERBATIM_MESSAGES));
	return { reason: args.reason?.trim() ?? "", older, recent: messages.slice(older.length) };
}

/** What one form of a seed holds, and what it leaves out. */
interface SeedForm {
	summary?: string;
	/** Older messages whose summary failed. */
	unsummarised: number;
	/** Messages left out so the seed fits its request (oldest first; the older ones with a dropped summary). */
	omitted: number;
	recent: SeedMessage[];
}

function renderSeed(selection: SeedSelection, form: SeedForm): string {
	const header = prompt
		.render(seedTemplate, {
			reason: selection.reason ? seedText(selection.reason) : "",
			summary: form.summary ? seedText(form.summary) : "",
			older: selection.older.length,
			unsummarised: form.unsummarised,
			omitted: form.omitted,
		})
		.trim();
	const transcript = transcriptOf(form.recent);
	return `<inbox-history>\n${header}${transcript ? `\n\n${transcript}` : ""}\n</inbox-history>`;
}

/**
 * The newest message alone, whole: the summary dropped and every other message left out (`summarised`: whether
 * the full seed has a summary, whose older messages are then left out too).
 */
function loneForm(selection: SeedSelection, unsummarised: number, summarised: boolean): SeedForm {
	const newest = selection.recent.at(-1);
	return {
		unsummarised,
		omitted: Math.max(0, selection.recent.length - 1) + (summarised ? selection.older.length : 0),
		recent: newest ? [newest] : [],
	};
}

/** `form` with its one message's text cut by `cut` characters from the end (the cut says how many). */
function cutForm(form: SeedForm, cut: number): SeedForm {
	const recent = form.recent.map(message => ({
		...message,
		text: message.text.slice(0, message.text.length - cut),
		cut,
	}));
	return { ...form, recent };
}

/**
 * Forms no seed of the messages `messageIds` names in `entries` is shorter than, whatever its summary (none is
 * made yet): every seed either keeps the newest message whole, with at least the header (the first form), or cuts
 * it, at most down to nothing (the second). The smaller one, as a request counts them, bounds the seed from below.
 */
export function topicSeedFloor(args: {
	entries: readonly SessionEntry[];
	messageIds: readonly string[];
	reason?: string;
}): string[] {
	const selection = selectSeed(args);
	const newest = selection.recent.at(-1);
	const bare: SeedForm = { unsummarised: 0, omitted: 0, recent: newest ? [newest] : [] };
	const others = Math.max(0, selection.recent.length - 1);
	return [
		renderSeed(selection, bare),
		renderSeed(selection, cutForm({ ...bare, omitted: others }, newest?.text.length ?? 0)),
	];
}

/** A form's text and size, as a request counts it. */
interface Measured {
	text: string;
	tokens: number;
}

/**
 * A topic's seed: its full text, and shorter forms for a request it does not fit. Shortening leaves out the
 * oldest verbatim messages first (the newest stays), then the summary of the older ones, then cuts the newest
 * message's text; a line says how many messages were left out and the cut says how many characters. Forms are
 * compared as rendered: those lines can make a form larger than the one it shortens.
 */
export class TopicSeed implements TranscriptPreamble {
	readonly text: string;
	readonly #selection: SeedSelection;
	readonly #full: SeedForm;

	constructor(selection: SeedSelection, summary: string | undefined) {
		this.#selection = selection;
		const unsummarised = selection.older.length > 0 && !summary ? selection.older.length : 0;
		this.#full = { summary, unsummarised, omitted: 0, recent: selection.recent };
		this.text = renderSeed(selection, this.#full);
	}

	fit(maxTokens: number, count: (text: string) => number): { text: string; tokens: number } {
		const measured: Measured[] = [];
		const measure = (form: SeedForm): Measured => {
			const text = renderSeed(this.#selection, form);
			const result = { text, tokens: count(text) };
			measured.push(result);
			return result;
		};
		/** The first of `form(lo..hi)` (each holding less than the one before) that fits, by bisection. */
		const firstFitting = (lo: number, hi: number, form: (index: number) => SeedForm): Measured | undefined => {
			let found: Measured | undefined;
			while (lo <= hi) {
				const mid = Math.floor((lo + hi) / 2);
				const candidate = measure(form(mid));
				if (candidate.tokens <= maxTokens) {
					found = candidate;
					hi = mid - 1;
				} else {
					lo = mid + 1;
				}
			}
			return found;
		};
		const full = this.#full;
		const whole = measure(full);
		if (whole.tokens <= maxTokens) return whole;
		const recent = full.recent;
		const kept = firstFitting(1, recent.length - 1, left => ({ ...full, omitted: left, recent: recent.slice(left) }));
		if (kept) return kept;
		const lone = loneForm(this.#selection, full.unsummarised, full.summary !== undefined);
		const alone = measure(lone);
		if (alone.tokens <= maxTokens) return alone;
		const cut = firstFitting(1, lone.recent[0]?.text.length ?? 0, cut => cutForm(lone, cut));
		return (
			cut ?? measured.reduce((smallest, candidate) => (candidate.tokens < smallest.tokens ? candidate : smallest))
		);
	}
}

/**
 * The seed for the messages `messageIds` names in `entries` (file order;
 * ids that name no user or assistant text message are skipped).
 */
export async function buildTopicSeed(args: {
	entries: readonly SessionEntry[];
	messageIds: readonly string[];
	reason?: string;
	summarize: (transcript: string) => Promise<string | undefined>;
}): Promise<TopicSeed> {
	const selection = selectSeed(args);
	let summary: string | undefined;
	if (selection.older.length > 0) {
		try {
			summary =
				(await args.summarize(transcriptOf(selection.older).slice(-SUMMARY_INPUT_MAX_CHARS)))?.trim() || undefined;
		} catch (error) {
			// A replaced transcript cancels the seed; it is not a failed summary to fall back from.
			if (error instanceof StaleTranscriptError) throw error;
			logger.warn("topics: seed summary failed", { error: error instanceof Error ? error.message : String(error) });
		}
	}
	return new TopicSeed(selection, summary);
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
