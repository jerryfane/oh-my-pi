/**
 * The Inbox transcript a topics operation started in: the Inbox session's
 * transcript generation at admission. The session checks it where an effect
 * happens (a turn starting or queueing, a row being appended, an atomic
 * write under its lock), so nothing that started in one transcript lands in
 * another: `/new`, `/clear`, a fork, and any switch or reload each make a new one.
 */

/** A write or turn for an Inbox transcript that is no longer the open one. */
export class InboxChangedError extends Error {
	constructor() {
		super("The Inbox session changed before the change could be saved.");
		this.name = "InboxChangedError";
	}
}

/** A topics operation's originating Inbox transcript. */
export class InboxBinding {
	/**
	 * @param generation The Inbox session's transcript generation at admission.
	 * @param sessionId The Inbox session id at admission, for attribution.
	 * @param currentGeneration The Inbox session's transcript generation now.
	 */
	constructor(
		readonly generation: number,
		readonly sessionId: string,
		private readonly currentGeneration: () => number,
	) {}

	/** Whether the bound transcript is still the open one, for work outside the session (a summary request). */
	get isCurrent(): boolean {
		return this.currentGeneration() === this.generation;
	}
}
