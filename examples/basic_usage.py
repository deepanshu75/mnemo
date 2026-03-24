from mnemo import MemoryStore

mem = MemoryStore(path="./memory.db")
mem.add("The deployment uses Docker on port 8080")
mem.add("Auth is handled via JWT, tokens expire in 24h")

results = mem.search("how does authentication work?", top_k=3)
for row in results:
    print(row.text, row.score)

context = mem.context(query="deployment setup", top_k=5)
print(context)
