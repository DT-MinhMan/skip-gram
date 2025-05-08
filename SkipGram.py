import numpy as np
import re
from collections import Counter
from pyvi import ViTokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Tiền xử lý dữ liệu:
# - chuyển hóa thành chữ thường
# - loại bỏ các kí tự đặt biệt (chỉ giữ lại các từ và khoảng trắng)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Tách các câu thành các từ (token), sau đó tách các từ đó thành danh sách
def tokenize_text(text):
    return ViTokenizer.tokenize(text).split()

# Xây dựng từ điển từ vựng
def build_vocabulary(corpus):
    word_counts = Counter()
    for sentence in corpus:
        word_counts.update(sentence)
    word_to_id = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    return word_to_id, id_to_word

# Tạo dữ liệu huấn luyện cho mô hình SkipGram
def generate_skipgram_data(corpus, word_to_id, window_size=1):
    data = []
    for sentence in corpus:
        sentence_ids = [word_to_id[word] for word in sentence if word in word_to_id]
        if not sentence_ids:
            continue
        for i, word_id in enumerate(sentence_ids):
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, len(sentence_ids))
            for j in range(start, end):
                if i != j:
                    data.append((word_id, sentence_ids[j]))
    return np.array(data)

# Khởi tạo mô hình SkipGram với các tham số:
# vocab_size: Kích thước từ vựng (số lượng từ trong dữ liệu).
# embedding_dim: Kích thước của vector nhúng (embedding).
# W: Ma trận trọng số cho các từ (ma trận này được huấn luyện).
# W_prime: Ma trận trọng số cho các ngữ cảnh.
class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W = np.random.uniform(0, 1, (self.vocab_size, embedding_dim))
        self.W_prime = np.random.uniform(0, 1, (embedding_dim, self.vocab_size))

# Tính toán xác suất cho các từ ngữ cảnh, dựa trên từ trung tâm:
# h: Vector nhúng của từ trung tâm.
# u: Sự kết hợp của h với các trọng số ngữ cảnh.
# y_pred: Xác suất ngữ cảnh theo phân phối softmax.
    def forward(self, center_word):
        h = self.W[center_word]
        u = np.dot(h, self.W_prime)
        y_pred = self.softmax(u)
        return y_pred, h, u

    #Tính softmax
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

# Tính toán gradient và cập nhật các trọng số:
# y: Mảng xác suất thật (một-hot vector).
# error: Lỗi giữa xác suất dự đoán và xác suất thật.
# Cập nhật W và W_prime dựa trên gradient.
    def backward(self, center_word, context_word, y_pred, learning_rate):
        y = np.zeros(self.vocab_size)
        y[context_word] = 1
        error = y_pred - y
        dW_prime = np.outer(self.W[center_word], error)
        dW = np.dot(self.W_prime, error)
        self.W[center_word] -= learning_rate * dW
        self.W_prime -= learning_rate * dW_prime

    # Ở đây lựa chọn batch size = 1 nên không thêm vào để  giữ cho mã code đơn giản
    def train(self, data, epochs, learning_rate):
        losses = []  # Danh sách để lưu trữ mất mát theo thời gian
        for epoch in range(epochs):
            loss = 0
            for center_word, context_word in data:
                y_pred, h, u = self.forward(center_word)
                self.backward(center_word, context_word, y_pred, learning_rate)
                # Tính toán loss bằng cross_entropy và cập nhật trọng số
                loss -= np.log(y_pred[context_word])
            normalized_loss = loss / len(data)
            losses.append(normalized_loss)  # Thêm giá trị mất mát vào danh sách
            print(f"Epoch {epoch + 1}, Loss: {normalized_loss:.4f}")

        # Vẽ biểu đồ giảm mất mát
        plt.plot(range(epochs), losses, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()
        plt.show()

    def get_word_vector(self, word, word_to_id):
        if word in word_to_id:
            return self.W[word_to_id[word]]
        raise ValueError(f"Word '{word}' not in vocabulary.")

    # Tính độ tương đồng cosine giữa hai từ
    def cosine_similarity(self, word1, word2, word_to_id):
        if word1 not in word_to_id or word2 not in word_to_id:
            raise ValueError(f"One or both words are not in the vocabulary: '{word1}', '{word2}'")
        vec1 = self.get_word_vector(word1, word_to_id)
        vec2 = self.get_word_vector(word2, word_to_id)
        dot_product = np.dot(vec1, vec2)
        return dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


if __name__ == "__main__":
    try:
        # Đọc dữ liệu từ file data_vietnamese.txt
        with open("data_vietnamese.txt", "r", encoding="utf-8") as f:
            raw_data = f.readlines()
    except FileNotFoundError:
        print("File not found: data_vietnamese.txt")
        exit(1)

    # Tiền xử lý
    processed_data = [tokenize_text(preprocess_text(line)) for line in raw_data]
    word_to_id, id_to_word = build_vocabulary(processed_data)
    skipgram_data = generate_skipgram_data(processed_data, word_to_id)

    vocab_size = len(word_to_id)
    embedding_dim = 100
    skipgram_model = SkipGramModel(vocab_size, embedding_dim)
    skipgram_model.train(skipgram_data, epochs=200, learning_rate=0.01)

    # Trực quan hóa mô hình
    embeddings = skipgram_model.W
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    max_words_to_plot = 50  # Giới hạn số từ được vẽ
    count = 0  # Đếm số từ đã vẽ

    for i, word in id_to_word.items():
        if i < len(reduced_embeddings):
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
            plt.text(reduced_embeddings[i, 0] + 0.01, reduced_embeddings[i, 1] + 0.01, word, fontsize=9)
            count += 1
        if count >= max_words_to_plot:
            break  # Dừng lại nếu đã vẽ đủ 50 từ
    plt.show()

    #Kiểm tra tương đồng giữa cặp từ vựng
    word1 = "học"
    word2 = "giáo_dục"
    word4 = "ăn_uống"
    similarity = skipgram_model.cosine_similarity(word1, word2, word_to_id)
    similarity1 = skipgram_model.cosine_similarity(word1, word4, word_to_id)
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")
    print(f"Cosine similarity between '{word1}' and '{word4}': {similarity1:.4f}")

