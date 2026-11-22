/**
 * Which transcript an {@link AgentSession} has open, for work bound to one
 * transcript (topics: notes, state writes, the Organise turn). The generation
 * changes in the same step that installs a replacement (`/new`, `/clear`, a
 * fork, `/btw` promotion, `/tree` navigation, any switch or reload) and is
 * restored when that replacement rolls back. While a transition is in progress
 * the identity is provisional: bound work is held until it settles, or refused
 * where it cannot wait.
 */
import { InboxChangedError } from "../topics/binding";

/** The session's transition barrier. */
export interface TranscriptTransitions {
	readonly isSessionTransitioning: boolean;
	waitForSessionTransition(): Promise<void>;
}

export class TranscriptIdentity {
	#generation = 0;
	readonly #transitions: TranscriptTransitions;

	constructor(transitions: TranscriptTransitions) {
		this.#transitions = transitions;
	}

	get generation(): number {
		return this.#generation;
	}

	/**
	 * A replacement is installed now: call in the same step as the install. Returns the rollback, which
	 * restores the previous identity when the install did not happen after all.
	 */
	replace(): () => void {
		const previous = this.#generation++;
		return () => {
			this.#generation = previous;
		};
	}

	/** Admit work bound to `expected` (none = unbound): hold through a transition, then refuse another transcript. */
	async admit(expected: number | undefined): Promise<void> {
		if (expected === undefined) return;
		while (this.#transitions.isSessionTransitioning) await this.#transitions.waitForSessionTransition();
		this.check(expected);
	}

	/** At the point of effect (synchronous): refuse a transcript being replaced, or another one. */
	check(expected: number | undefined): void {
		if (expected === undefined) return;
		if (this.#transitions.isSessionTransitioning || expected !== this.#generation) throw new InboxChangedError();
	}

	/**
	 * For the session's own boundary flush, which runs before any install even inside a transition: whether
	 * work bound to `expected` still belongs to the installed transcript.
	 */
	matches(expected: number | undefined): boolean {
		return expected === undefined || expected === this.#generation;
	}
}
