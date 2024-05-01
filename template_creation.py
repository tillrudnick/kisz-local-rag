# from langchain.schema.vectorstore import VectorStore
#
# def create_prompt_template(db: VectorStore, num_examples: int = NUM_EXAMPLES_IN_PROMPT) -> ChatMLPromptTemplate:
#     example_selector = MaxMarginalRelevanceExampleSelector(vectorstore=db, k=num_examples)
#     examples_prompt = FewShotChatMessagePromptTemplate(
#         # Which input variable(s) will be passed to the example selector.
#         input_variables=["example_query"],
#         example_selector=example_selector,
#         # Define how each example will be formatted.
#         # In this case, each example will become 2 messages:
#         # 1 human, and 1 AI
#         example_prompt=(
#                 HumanMessagePromptTemplate.from_template(f"{INPUT_PROMPT}:\n{{input_text}}")
#                 + AIMessagePromptTemplate.from_template(f"{TARGET_PROMPT}:\n{{target_text}}")
#         ),
#     )
#     # Define the overall prompt.
#     prompt = ChatMLPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
#         examples_prompt,
#         HumanMessagePromptTemplate.from_template(f"{INPUT_PROMPT}:\n{{input_text}}"),
#         PartialAIMessagePromptTemplate.from_template(f"{TARGET_PROMPT}:\n")
#     ])
#     return prompt
