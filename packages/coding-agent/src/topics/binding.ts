/**
 * The Inbox transcript a topics operation started in. Every mutating topics
 * operation (a suggestion, an accept, a dismissal, Organise, a seeded topic)
 * binds at admission and checks the binding before each read it acts on and
 * each write it makes, so nothing that started in one Inbox transcript lands
 * in another: a `/new`, a `/clear`, a fork, a switch and a reopen of an
 * earlier session all make a different transcript.
 */

/** A write or read for an Inbox transcript that is no longer the open one. */
export class InboxChangedError extends Error {
	constructor() {
		super("The Inbox session changed before the change could be saved.");
		this.name = "InboxChangedError";
	}
}

/** The identity of one Inbox transcript (see {@link InboxBinding.key}). */
export interface InboxIdentity {
	sessionId: string;
	sessionFile: string | undefined;
	/** Counts every change of the Inbox's session (a new, forked, switched or reopened one). */
	sessionEpoch: number;
	/** Counts transcript replacements that keep the session (`/clear`). */
	transcriptEpoch: number;
}

/** A topics operation's originating Inbox transcript. */
export class InboxBinding {
	/** Unique per transcript: equal keys mean the same open transcript. */
	readonly key: string;
	readonly #current: () => InboxIdentity;

	constructor(current: () => InboxIdentity) {
		this.#current = current;
		this.key = inboxKey(current());
	}

	/** Whether the bound transcript is still the open one. */
	get isCurrent(): boolean {
		return inboxKey(this.#current()) === this.key;
	}

	/** Throw {@link InboxChangedError} unless the bound transcript is still the open one. */
	assertCurrent(): void {
		if (!this.isCurrent) throw new InboxChangedError();
	}
}

function inboxKey({ sessionId, sessionFile, sessionEpoch, transcriptEpoch }: InboxIdentity): string {
	return JSON.stringify([sessionId, sessionFile ?? null, sessionEpoch, transcriptEpoch]);
}
