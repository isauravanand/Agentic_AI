I want to make a multi agent system
first i want to make a single agent system
it will have one llm (groq)
it will haw 2 tools
one web search using tavily. you will find all api keys in env file
another will be a pyton calculator   there will be one ask llm function
llm will decide which tiools to uyse and how many times.
there will be a max attempts o 5   then we will convcert this to a multi aghent sstem without langraph   in the muti agent there will be three ask llm functions
first will be of manager llm
its ponly job is to decide which helper llm to use
second will be ask_search_llm which has access to only th search tool
third will be the ask_maths_llm which will have access to call tool   the role of each ask llm can be give n by a syteme prompt
manager llm will also have a notes file where it will keep trakc of the reponses by the helper llm   finally we will have token usage uin this also
in the end we will compare botht the systems and see which used more tokens
for test prepare a little complex query which incolves seacrh and claucltort (maybe more than once)   if you have doubts then ask 