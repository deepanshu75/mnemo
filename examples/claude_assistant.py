import anthropic

from mnemo.integrations.claude import ClaudeMemory

mem = ClaudeMemory(path="./memory.db")
client = anthropic.Anthropic()

mem.add("User prefers responses in bullet points. Timezone is CET.")

messages = mem.build_messages(
    query="summarise our project status",
    user_message="what did we decide about the auth flow?",
)

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=messages,
)

print(response)
