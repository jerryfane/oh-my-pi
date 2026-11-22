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

	/** The current epoch; throws {@link StaleTranscriptError} at once while a replacement runs. */
	capture(): InboxEpoch {
		const current = this.#current;
		if (!current) throw new StaleTranscriptError();
		return { id: current.id, signal: current.controller.signal };
	}

	/** A replacement begins: abort the current epoch (synchronously) and leave none. */
	end(): void {
		this.#current?.controller.abort(new StaleTranscriptError());
		this.#current = undefined;
	}

	/** The replacement settled: start a new epoch unless one is current (never revives an ended one). */
	start(): void {
		this.#current ??= { id: this.#nextId++, controller: new AbortController() };
	}
}
