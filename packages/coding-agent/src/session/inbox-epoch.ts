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
 * Admit a transcript-bound prompt at its last check before dispatch: put `transcriptPreamble` ahead of the prompt's
 * text, in the prompt itself (`message`, or its prepared form, the last user message of `messages`), then report the
 * admission. Only here does the preamble join the prompt: no setup step before it reads the preamble.
 */
export function admitBoundPrompt(
	messages: AgentMessage[],
	message: AgentMessage,
	options: { transcriptPreamble?: string; onPromptAdmitted?: () => void },
): void {
	const preamble = options.transcriptPreamble;
	if (preamble) {
		const index = messages.includes(message)
			? messages.indexOf(message)
			: messages.findLastIndex(m => m.role === "user");
		const prompt = messages[index];
		if (prompt?.role !== "user") throw new Error("A transcript-bound prompt with a preamble must be a user message.");
		// In place: the prompt keeps its identity (turn observation, queued-message bookkeeping).
		prompt.content = withPreamble(prompt.content, preamble);
	}
	options.onPromptAdmitted?.();
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
