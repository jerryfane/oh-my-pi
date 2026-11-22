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

import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import { invalidateMessageCache } from "@oh-my-pi/pi-agent-core/compaction";
import type { UserMessage } from "@oh-my-pi/pi-ai";

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
 * What pre-prompt context maintenance sizes for a transcript-bound prompt: `messages` with `message` (the prompt)
 * replaced by a copy that already carries the preamble, so the seed counts toward compaction and context promotion.
 * The copy is only counted, never sent or stored: no auxiliary call (judge, summary) reads its text. Without a
 * preamble, `messages` itself.
 */
export function withBoundPreamble(
	messages: AgentMessage[],
	message: AgentMessage,
	{ transcriptPreamble: preamble }: { transcriptPreamble?: string } = {},
): AgentMessage[] {
	if (!preamble) return messages;
	const prompt = boundPrompt(message);
	return messages.map(candidate =>
		candidate === message ? { ...prompt, content: withPreamble(prompt.content, preamble) } : candidate,
	);
}

/**
 * Admit a transcript-bound prompt at its last check before dispatch: put `transcriptPreamble` ahead of the text
 * of `message` (the exact prompt the batch carries), then report the admission. Only here does the preamble join
 * the prompt: no setup step before it reads the preamble. In place, so the prompt keeps its identity (turn
 * observation, queued-message bookkeeping); its cached token estimate and conversion are invalidated with it.
 */
export function admitBoundPrompt(
	message: AgentMessage,
	options: { transcriptPreamble?: string; onPromptAdmitted?: () => void },
): void {
	const preamble = options.transcriptPreamble;
	if (preamble) {
		const prompt = boundPrompt(message);
		prompt.content = withPreamble(prompt.content, preamble);
		invalidateMessageCache(prompt);
	}
	options.onPromptAdmitted?.();
}

function boundPrompt(message: AgentMessage): UserMessage {
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
