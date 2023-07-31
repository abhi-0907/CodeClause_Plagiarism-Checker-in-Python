import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

txt_names = [doc for doc in os.listdir() if doc.endswith('.txt')]
txts = [open(_file, encoding='utf-8').read()
                 for _file in txt_names]


def vectorize(Text): 
    return TfidfVectorizer().fit_transform(Text).toarray()
def similarity(doc1, doc2): 
    return cosine_similarity([doc1, doc2])


vectorized_txts = vectorize(txts)
txt_vectors = list(zip(txt_names, vectorized_txts))
plagiarism_results = set()

for txt_name1, txt_vector1 in txt_vectors:
    new_vectors = txt_vectors.copy()
    current_index = new_vectors.index((txt_name1, txt_vector1))
    del new_vectors[current_index]
    for txt_name2, txt_vector2 in new_vectors:
        sim_score = similarity(txt_vector1, txt_vector2)[0][1]
        sim_score1 = similarity(txt_vector1, txt_vector2)
        txt_pair = sorted((txt_name1, txt_name2))
        score = (txt_pair[0], txt_pair[1], sim_score)
        plagiarism_results.add(score)


for data in plagiarism_results:
    print(data)