# Annotation guidelines (binary label `relation_present`)

**Task.** Given a Hebrew passage, a subject entity, a Wikidata predicate and an object entity: does the passage explicitly or clearly implicitly express the stated relation between the subject and the object?

**Label 1 (relation present)** when
- the passage contains a sentence or clause that directly states the relation (e.g. "X was born in Y" for *place of birth*), or
- the relation can be unambiguously inferred within a single sentence or adjacent sentences, without external knowledge.

**Label 0 (relation absent)** when
- the passage mentions both entities but does not express the stated relation between them;
- the passage expresses a related but distinct relation (e.g. *lives in* rather than *was born in*);
- the relation can only be inferred by combining distant passages or requires external knowledge;
- the passage is ambiguous as to which of two possible objects the predicate applies.

**Borderline cases.** Default to 0 and flag the example for adjudication when the predicate definition itself is unclear.

Both gold sets were annotated independently by two native Hebrew speakers with NLP expertise following these guidelines; disagreements were adjudicated and the adjudicated label is the `label` column.
