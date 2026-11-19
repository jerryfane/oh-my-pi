<irc>
{{label}}:

{{message}}
{{#if relay}}

Your answer goes back to `{{from}}` when this turn ends; for an interim note write `agent://{{from}}?progress`.
{{/if}}
{{#unless note}}

Another agent started this turn, so in it you cannot ask other topics.
{{/unless}}
</irc>
