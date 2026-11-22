/**
 * Inbox epochs: one per transcript an {@link AgentSession} has open, for work
 * bound to that transcript (topics: notes, state writes, the Organise turn, a
 * topic's seed). When a replacement begins (`/new`, `/clear`, a switch or
 * reload, a fork, a branch, `/btw` promotion, `/tree` navigation that moves
 * the branch) the epoch is aborted at once, before anything awaits; while the
 * replacement runs there is none, and a new one starts only once it settled,
 * whichever transcript it settled on. Bound work carries the epoch's `signal`
 * and stops wherever it finds it aborted.
 */

import type { AgentMessage, Tokenizer } from "@oh-my-pi/pi-agent-core";
import { invalidateMessageCache, promptBudgetTokens } from "@oh-my-pi/pi-agent-core/compaction";
import type { AssistantMessage, Model, UserMessage } from "@oh-my-pi/pi-ai";
import * as AIError from "@oh-my-pi/pi-ai/error";
import { isMeaningfulCompletionEvent } from "@oh-my-pi/pi-ai/utils/empty-completion-retry";
import type { Settings } from "../config/settings";
import type { AgentSessionEvent } from "./agent-session-events";
import { cfgContextPromotionEnabled } from "./context-settings";
import { resolveModelCompactionSettings } from "./model-compaction-threshold";
import { resolveContextPromotionConfiguredTarget } from "./role-models";

/** Work bound to a transcript that was replaced, or captured while a replacement runs. */
export class StaleTranscriptError extends Error {
	constructor() {
		super("The Inbox session changed before the change could be saved.");
		this.name = "StaleTranscriptError";
	}
}

/** One transcript's epoch: `signal` aborts when a replacement of that transcript begins. */
export interface InboxEpoch {
	readonly id: number;
	readonly signal: AbortSignal;
}

/** Throw {@link StaleTranscriptError} when `signal` (bound work's epoch) is aborted. */
export function throwIfStale(signal: AbortSignal | undefined): void {
	if (signal?.aborted) throw new StaleTranscriptError();
}

/** The session's epochs: at most one current, none while a replacement runs. */
export class InboxEpochs {
	#nextId = 1;
	#current: { id: number; controller: AbortController } | undefined = { id: 0, controller: new AbortController() };

	/** Queued messages bound to an epoch's signal, by message. */
	readonly #boundMessages = new WeakMap<object, AbortSignal>();

	readonly #onEnd: () => void;

	/** `onEnd` runs once each time an epoch ends, after its signal aborted: one sweep of what was bound to it. */
	constructor(onEnd: () => void) {
		this.#onEnd = onEnd;
	}

	/** The current epoch; throws {@link StaleTranscriptError} at once while a replacement runs. */
	capture(): InboxEpoch {
		const current = this.#current;
		if (!current) throw new StaleTranscriptError();
		return { id: current.id, signal: current.controller.signal };
	}

	/** A replacement begins: abort the current epoch (synchronously) and leave none. */
	end(): void {
		const ended = this.#current;
		if (!ended) return;
		this.#current = undefined;
		ended.controller.abort(new StaleTranscriptError());
		this.#onEnd();
	}

	/** The replacement settled: start a new epoch unless one is current (never revives an ended one). */
	start(): void {
		this.#current ??= { id: this.#nextId++, controller: new AbortController() };
	}

	/** Bind `messages` (one queued batch) to `signal`, its epoch's: once that ends, {@link isLive} is false for them. */
	bindQueued(messages: readonly object[], signal: AbortSignal): void {
		for (const message of messages) this.#boundMessages.set(message, signal);
	}

	/** Whether a queued `message` may still be delivered: unbound, or its transcript was not replaced. */
	readonly isLive = (message: object): boolean => this.#boundMessages.get(message)?.aborted !== true;
}

/**
 * Text put ahead of a transcript-bound prompt when it is admitted (a topic's seed): its full `text`, and how it
 * shortens to fit a request.
 */
export interface TranscriptPreamble {
	readonly text: string;
	/**
	 * The longest form of the preamble within `maxTokens` (as `count` measures text), the whole one first, and its
	 * `tokens`; when no form fits, the one `count` finds smallest.
	 */
	fit(maxTokens: number, count: (text: string) => number): { text: string; tokens: number };
}

/** A transcript-bound prompt refused at admission: even its shortest preamble makes the request too large. */
export class PreambleTooLargeError extends Error {
	constructor(
		readonly tokens: number,
		readonly limit: number,
	) {
		super(`The prompt's copied history is too large for the model: ${tokens} tokens, limit ${limit}.`);
		this.name = "PreambleTooLargeError";
	}
}

/** Why a launch's first request did not start: too large for its model (`seed_too_large`), or anything else. */
export type StartFailedCode = "seed_too_large" | "start_failed";

/**
 * How a launch's first request went: `started` once the provider answered it with content; `start-failed` when its
 * turn ended before that, with the `error` and, for a size cause, the request's `tokens` and the `limit` it exceeded
 * where known; `cancelled` when the launch was aborted first.
 */
export type FirstTurnOutcome =
	| { kind: "started" }
	| { kind: "start-failed"; code: StartFailedCode; error: string; tokens?: number; limit?: number }
	| { kind: "cancelled" };

/** What {@link watchFirstTurn} reads of a session: its model and its events. */
export interface FirstTurnSession {
	readonly model: Model | undefined;
	subscribe(listener: (event: AgentSessionEvent) => void): () => void;
}

/**
 * Watch `session`'s first turn and `report` how its request went, once. It started when the provider answered with
 * content: a streamed event that commits a provider attempt ({@link isMeaningfulCompletionEvent}: text, thinking or
 * tool-call argument content, not a block's opening marker), or an answer that ends without an error. Retries inside
 * the turn (compaction or context promotion after an overflow) count: a later answer still starts it. The returned
 * function takes the turn and reports when it settles unstarted: `cancelled` once `signal` (the run's) aborted,
 * else `start-failed` ({@link startFailure}). Without `report`, nothing is watched.
 */
export function watchFirstTurn(
	session: FirstTurnSession,
	signal: AbortSignal,
	report: ((outcome: FirstTurnOutcome) => void) | undefined,
): <T>(turn: Promise<T>) => Promise<T> {
	if (!report) return turn => turn;
	let started = false;
	let failure: AssistantMessage | undefined;
	const unsubscribe = session.subscribe(event => {
		if (started || (event.type !== "message_update" && event.type !== "message_end")) return;
		if (event.message.role !== "assistant") return;
		if (event.type === "message_update") {
			started = isMeaningfulCompletionEvent(event.assistantMessageEvent);
		} else if (event.message.stopReason === "error" || event.message.stopReason === "aborted") {
			failure = event.message;
		} else {
			started = true;
		}
		if (!started) return;
		unsubscribe();
		report({ kind: "started" });
	});
	const settle = (error: unknown) => {
		unsubscribe();
		if (started) return;
		report(signal.aborted ? { kind: "cancelled" } : startFailure(error, failure, session.model));
	};
	return async turn => {
		try {
			const value = await turn;
			settle(undefined);
			return value;
		} catch (error) {
			settle(error);
			throw error;
		}
	};
}

/**
 * Why a first turn did not start, from what it threw (`error`) or its last failed answer (`failure`): a preamble
 * that did not fit, or a provider that refused the request as too long for `model`, is `seed_too_large`.
 */
function startFailure(
	error: unknown,
	failure: AssistantMessage | undefined,
	model: Model | undefined,
): FirstTurnOutcome {
	if (error instanceof PreambleTooLargeError) {
		return {
			kind: "start-failed",
			code: "seed_too_large",
			error: error.message,
			tokens: error.tokens,
			limit: error.limit,
		};
	}
	const contextWindow = model?.contextWindow ?? 0;
	if (error === undefined && failure && AIError.isContextOverflow(failure, contextWindow)) {
		return {
			kind: "start-failed",
			code: "seed_too_large",
			error: failure.errorMessage ?? "The request is too long for the model.",
			...(contextWindow > 0 ? { limit: contextWindow } : {}),
		};
	}
	const reason = error instanceof Error ? error.message : error === undefined ? failure?.errorMessage : String(error);
	return {
		kind: "start-failed",
		code: "start_failed",
		error: reason || "The first request ended before the provider answered.",
	};
}

/** The models and windows a session runs with (its registry fits each window to its settings). */
export interface PromptModels {
	getAvailable(): Model[];
	fitContextWindow(model: Model, settings: Settings): Model;
}

/** Tokens a request on `model` may hold: its window less the output reserve (no window known: none). */
export function promptLimit(model: Model, settings: Settings): number | undefined {
	const contextWindow = model.contextWindow ?? 0;
	if (contextWindow <= 0) return undefined;
	return promptBudgetTokens(contextWindow, resolveModelCompactionSettings(settings, model));
}

/** The larger model a request on `model` is promoted to when it does not fit (compaction and promotion on). */
export function promotionTarget(model: Model, settings: Settings, models: PromptModels): Model | undefined {
	if (!resolveModelCompactionSettings(settings, model).enabled || !cfgContextPromotionEnabled.get(settings)) return;
	const target = resolveContextPromotionConfiguredTarget(model, models.getAvailable());
	return target && models.fitContextWindow(target, settings);
}

/** What a bound prompt reads of its session: its model (after maintenance), settings, models and context count. */
export interface BoundPromptSession {
	readonly model: Model | undefined;
	readonly settings: Settings;
	readonly modelRegistry: PromptModels;
	readonly agent: { readonly tokenizer: Tokenizer };
	getContextBreakdown(options: {
		contextWindow: number;
		pendingMessages: AgentMessage[];
	}): { usedTokens: number } | undefined;
}

/** The prompt's content with a preamble fitted to a request, the request's size, and whether it fits. */
interface Fitted {
	content: UserMessage["content"];
	tokens: number;
	limit: number;
	fits: boolean;
}

/**
 * A transcript-bound prompt (`transcriptSignal`) of `session`: the transcript's signal, the exact prompt the batch
 * carries, and its preamble, fitted to the request: whole when the request fits the model's limit (its window less
 * the output reserve), else shortened ({@link TranscriptPreamble.fit}). The content with the whole preamble is built
 * once, and a shortened one once per room it was fitted to. Pre-prompt maintenance sizes the batch with the preamble
 * fitted to the largest model the request can be promoted to ({@link sized}); admission, the last check before
 * dispatch, fits it to the model the request goes to and puts it in the prompt ({@link admit}). Only then does the
 * preamble join the prompt: no setup step before it reads it.
 */
export class BoundPrompt {
	readonly #signal: AbortSignal;
	readonly #prompt: AgentMessage;
	readonly #session: BoundPromptSession;
	readonly #preamble: TranscriptPreamble | undefined;
	readonly #onAdmitted: (() => void) | undefined;
	#whole: UserMessage["content"] | undefined;
	#shortened: { room: number; text: string; tokens: number; content: UserMessage["content"] } | undefined;

	private constructor(
		signal: AbortSignal,
		prompt: AgentMessage,
		session: BoundPromptSession,
		options: { transcriptPreamble?: TranscriptPreamble; onPromptAdmitted?: () => void },
	) {
		this.#signal = signal;
		this.#prompt = prompt;
		this.#session = session;
		this.#preamble = options.transcriptPreamble;
		if (this.#preamble) userPrompt(prompt);
		this.#onAdmitted = options.onPromptAdmitted;
	}

	/** `message` of `session`, bound by `options.transcriptSignal`; undefined for an unbound prompt. */
	static of(
		message: AgentMessage,
		options:
			| { transcriptSignal?: AbortSignal; transcriptPreamble?: TranscriptPreamble; onPromptAdmitted?: () => void }
			| undefined,
		session: BoundPromptSession,
	): BoundPrompt | undefined {
		const signal = options?.transcriptSignal;
		return signal ? new BoundPrompt(signal, message, session, options) : undefined;
	}

	/**
	 * `messages` as pre-prompt maintenance sizes them: the prompt replaced by a copy carrying the preamble fitted to
	 * the largest model the request can be promoted to, so it counts toward compaction and context promotion as it
	 * will be sent. The copy is only counted, never sent or stored: no auxiliary call (judge, summary) reads its text.
	 */
	sized(messages: AgentMessage[]): AgentMessage[] {
		if (!this.#preamble) return messages;
		const { model, settings, modelRegistry } = this.#session;
		const target = model && promotionTarget(model, settings, modelRegistry);
		const limit = model && promptLimit(model, settings);
		const targetLimit = target && promptLimit(target, settings);
		const { content } = this.#fitted(messages, limit === undefined ? undefined : Math.max(limit, targetLimit ?? 0));
		const prompt = userPrompt(this.#prompt);
		return messages.map(candidate => (candidate === prompt ? { ...prompt, content } : candidate));
	}

	/**
	 * The last check before dispatch of `messages` (the final batch, its notices included, with the bare prompt):
	 * false when the transcript was replaced (nothing is sent). Else the preamble, fitted to the model the request
	 * goes to, joins the prompt in place, so the prompt keeps its identity (turn observation, queued-message
	 * bookkeeping), with its cached token estimate and conversion invalidated; then the prompt is reported admitted.
	 * When even the shortest preamble does not fit, the prompt is refused ({@link PreambleTooLargeError}).
	 * `deliver` is told whether the batch was admitted, so the notices it carries are marked delivered only then.
	 */
	admit(messages: AgentMessage[], deliver?: (admitted: boolean) => void): boolean {
		let admitted = false;
		try {
			admitted = this.#admit(messages);
			return admitted;
		} finally {
			deliver?.(admitted);
		}
	}

	#admit(messages: AgentMessage[]): boolean {
		if (this.#signal.aborted) return false;
		if (this.#preamble) {
			const { model, settings } = this.#session;
			const fitted = this.#fitted(messages, model && promptLimit(model, settings));
			if (!fitted.fits) throw new PreambleTooLargeError(fitted.tokens, fitted.limit);
			const prompt = userPrompt(this.#prompt);
			prompt.content = fitted.content;
			invalidateMessageCache(prompt);
		}
		this.#onAdmitted?.();
		return true;
	}

	/**
	 * The prompt's content with the preamble fitted to a request of `messages` (with the bare prompt) within `limit`
	 * tokens (none known: the whole preamble). The prompt with the preamble is counted as a message, so separators
	 * and message framing are part of the fit.
	 */
	#fitted(messages: AgentMessage[], limit: number | undefined): Fitted {
		const prompt = userPrompt(this.#prompt);
		this.#whole ??= withPreamble(prompt.content, this.#preamble!.text);
		if (limit === undefined) return { content: this.#whole, tokens: 0, limit: 0, fits: true };
		const { tokenizer } = this.#session.agent;
		const contextWindow = this.#session.model?.contextWindow ?? limit;
		const rest = this.#session.getContextBreakdown({ contextWindow, pendingMessages: messages })?.usedTokens ?? 0;
		const bare = tokenizer.countMessage({ ...prompt });
		/** The request's size with `content` as the prompt's. */
		const size = (content: UserMessage["content"]) => rest - bare + tokenizer.countMessage({ ...prompt, content });
		const whole = size(this.#whole);
		if (whole <= limit) return { content: this.#whole, tokens: whole, limit, fits: true };
		const count = (text: string) => tokenizer.countTokens(text);
		// The preamble's own count leaves out its framing in the message: shrink the room by what that adds, and refit.
		let room = limit - rest;
		for (let attempt = 0; attempt < 3; attempt++) {
			const shortened = this.#shorten(room, count);
			const tokens = size(shortened.content);
			if (tokens <= limit) return { content: shortened.content, tokens, limit, fits: true };
			if (shortened.tokens > room) return { content: shortened.content, tokens, limit, fits: false };
			room -= tokens - limit;
		}
		const shortest = this.#shorten(0, count);
		const tokens = size(shortest.content);
		return { content: shortest.content, tokens, limit, fits: tokens <= limit };
	}

	/** The prompt's content with the preamble shortened to `room` tokens, built once per room. */
	#shorten(room: number, count: (text: string) => number) {
		if (this.#shortened?.room === room) return this.#shortened;
		const { text, tokens } = this.#preamble!.fit(room, count);
		const content =
			this.#shortened?.text === text
				? this.#shortened.content
				: withPreamble(userPrompt(this.#prompt).content, text);
		this.#shortened = { room, text, tokens, content };
		return this.#shortened;
	}
}

function userPrompt(message: AgentMessage): UserMessage {
	if (message.role !== "user") throw new Error("A transcript-bound prompt with a preamble must be a user message.");
	return message;
}

/** `content` with `preamble` ahead of its text, joined to its first text block as one text. */
function withPreamble(content: UserMessage["content"], preamble: string): UserMessage["content"] {
	if (typeof content === "string") return `${preamble}\n\n${content}`;
	const first = content.findIndex(block => block.type === "text");
	if (first < 0) return [{ type: "text", text: preamble }, ...content];
	return content.map((block, index) =>
		index === first && block.type === "text" ? { ...block, text: `${preamble}\n\n${block.text}` } : block,
	);
}
