import { register } from "../config/registry";

export const cfgTopicsEnabled = register({
	id: "topics.enabled",
	type: "boolean",
	default: false,
	ui: {
		tab: "tasks",
		group: "Modes",
		label: "Topics",
		description: "Keep owner-facing topic conversations as persistent workers next to the main session",
	},
});
