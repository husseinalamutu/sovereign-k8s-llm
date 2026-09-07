# Linguistic Evaluation Methodology: Yoruba Localization

## Background
Yoruba (`yo`) is an underrepresented, Niger-Congo tonal language spoken by over 50 million people primarily in Nigeria, Benin, and Togo. Standard commercial LLMs trained overwhelmingly on English often exhibit three distinct failure modes in Yoruba:
1. **Token Inefficiency:** Words with diacritics (`ẹ`, `ọ`, `ṣ`, tone marks `á`, `à`, `ā`) are split into multiple byte-level tokens, leading to rapid context consumption and higher generation latency.
2. **Code-Switching Drift:** When asked complex questions, models frequently drift back into English or generic calques rather than maintaining monolingual Yoruba grammar.
3. **Loss of Honorifics and Cultural Nuance:** Lack of contextual distinctions between formal/elder respect forms (`Ẹ`, `Wọn`) and informal singular forms (`O`, `Iwọ`).

## Evaluation Dimensions
Each evaluation prompt is scored across four qualitative dimensions:
1. **Tonal & Orthographic Accuracy (1–5):** Are diacritics and tone markings appropriately placed or approximated without garbled Unicode sequences?
2. **Grammatical Naturalness (1–5):** Does the text read like native Yoruba or a literal English transliteration?
3. **Cultural & Idiomatic Alignment (1–5):** Does the response correctly leverage Yoruba proverbs (*òwe*), metaphors, and honorific forms (*ọ̀wọ̀*)?
4. **Factual & Contextual Relevance (1–5):** Does the response actually answer the technical or advisory query correctly?

## Observed Delta: Base vs. Fine-Tuned
- **Base `tiny-aya-earth`:** Possesses foundational vocabulary and acknowledges Yoruba tokens, but frequently generates repetitive loop phrases or slips into code-mixed Nigerian Pidgin/English when prompted for detailed multi-sentence explanations.
- **Fine-Tuned with `african-ultrachat`:** Produces cohesive paragraphs in authentic Yoruba, retains honorific address (`Ẹ ku aaro o`), maintains topic consistency across multi-step instructions, and properly contextualizes modern concepts (like internet networking or agricultural methods) using culturally accessible metaphors.
