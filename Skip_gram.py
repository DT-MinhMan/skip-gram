import numpy as np
import random
from tqdm import tqdm
from pyvi import ViTokenizer
import re
import string
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class SkipGramModel:
    def __init__(self, corpus, embedding_dim=100, learning_rate=0.01, epochs=10, batch_size=1, window_size=2):
        """
        Khởi tạo mô hình Skip-Gram.
        Args:
            corpus (list): Danh sách câu đã tiền xử lý.
            embedding_dim (int): Kích thước vector nhúng.
            learning_rate (float): Tốc độ học.
            epochs (int): Số lần lặp huấn luyện.
            batch_size (int): Kích thước batch.
            window_size (int): Kích thước cửa sổ ngữ cảnh.
        """
        self.corpus = [self.preprocess_text(text) for text in corpus]
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.vocab, self.vocab_size = self.build_vocabulary()
        self.W1 = np.random.uniform(-1, 1, (self.vocab_size, embedding_dim))
        self.W2 = np.random.uniform(-1, 1, (embedding_dim, self.vocab_size))

    @staticmethod
    def preprocess_text(text):
        """
        Tiền xử lý văn bản:
        - Chuyển chữ thường
        - Loại bỏ dấu câu và ký tự đặc biệt
        - Tách từ bằng PyVi
        """
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", " ", text)  # Loại bỏ dấu câu
        text = re.sub(r"\s+", " ", text).strip()  # Loại bỏ khoảng trắng thừa
        return ViTokenizer.tokenize(text)

    def build_vocabulary(self):
        """Tạo từ điển từ vựng từ tập dữ liệu."""
        vocab = {}
        for sentence in self.corpus:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab, len(vocab)

    def softmax(self, x):
        """Tính softmax của vector đầu vào."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def cross_entropy_loss(self, predictions, target_index):
        """Tính cross-entropy loss giữa dự đoán và nhãn thực tế."""
        return -np.log(predictions[target_index])

    def generate_training_data(self):
        """Tạo dữ liệu huấn luyện từ tập văn bản."""
        training_data = []
        for sentence in self.corpus:
            words = sentence.split()
            word_indices = [self.vocab[word] for word in words if word in self.vocab]
            for center_idx in range(len(word_indices)):
                for offset in range(-self.window_size, self.window_size + 1):
                    context_idx = center_idx + offset
                    if 0 <= context_idx < len(word_indices) and center_idx != context_idx:
                        training_data.append((word_indices[center_idx], word_indices[context_idx]))
        return training_data

    def forward(self, center_word_idx):
        """
        Forward pass.
        Args:
            center_word_idx (int): Chỉ số từ trung tâm.
        Returns:
            h (ndarray): Vector nhúng của từ trung tâm.
            u (ndarray): Điểm số dự đoán cho các từ trong từ vựng.
            y_hat (ndarray): Xác suất dự đoán.
        """
        h = self.W1[center_word_idx]
        u = np.dot(h, self.W2)
        y_hat = self.softmax(u)
        return h, u, y_hat

    def backward(self, center_word_idx, context_word_idx, y_hat):
        """
        Backward pass và cập nhật trọng số.
        Args:
            center_word_idx (int): Chỉ số từ trung tâm.
            context_word_idx (int): Chỉ số từ trong ngữ cảnh.
            y_hat (ndarray): Xác suất dự đoán.
        """
        y = np.zeros(self.vocab_size)
        y[context_word_idx] = 1
        error = y_hat - y

        dW2 = np.outer(self.W1[center_word_idx], error)
        dW1 = np.dot(self.W2, error)

        self.W1[center_word_idx] -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2

    def cross_entropy_loss(self, y_hat, target_index):
        """
        Tính cross-entropy loss giữa dự đoán và nhãn thực tế.
        Args:
            y_hat (ndarray): Vector xác suất (sau softmax), kích thước [|V|].
            target_index (int): Chỉ số của từ trong ngữ cảnh.
        Returns:
            loss (float): Giá trị cross-entropy loss.
        """
        return -np.log(y_hat[target_index] + 1e-9)  # Thêm epsilon để tránh log(0)

    def train(self):
        """Huấn luyện mô hình Skip-Gram."""
        training_data = self.generate_training_data()
        for epoch in range(self.epochs):
            total_loss = 0
            for center_idx, context_idx in tqdm(training_data, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                # Forward pass
                h, u, y_hat = self.forward(center_idx)

                # Tính toán loss
                loss = self.cross_entropy_loss(y_hat, context_idx)
                total_loss += loss

                # Backward pass
                self.backward(center_idx, context_idx, y_hat)

            # In tổng loss sau mỗi epoch
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                save_model_pickle(self, 'skipgram_model.pkl')
                print(f"Model saved mode in epoch: {epoch + 1}")

    def cosine_similarity(self, vector1, vector2):
        """Tính toán cosine similarity giữa hai vector."""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def evaluate_similarity(self, word_pairs):
        """
        Đánh giá vector nhúng bằng cách kiểm tra độ tương đồng cosine giữa các từ.
        Args:
            word_pairs (list): Danh sách cặp từ [(word1, word2), ...].
        """
        for word1, word2 in word_pairs:
            if word1 in self.vocab and word2 in self.vocab:
                vec1 = self.W1[self.vocab[word1]]
                vec2 = self.W1[self.vocab[word2]]
                similarity = self.cosine_similarity(vec1, vec2)
                print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
            else:
                print(f"One or both words not in vocabulary: '{word1}', '{word2}'")



def save_model_pickle(model, filename):
    """
    Save the entire SkipGram model using pickle.
    Args:
        model (SkipGram): The trained SkipGram model.
        filename (str): The filename to save the model.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model_pickle(filename):
    """
    Load the entire SkipGram model from a file using pickle.
    Args:
        filename (str): The filename to load the model.
    Returns:
        SkipGram: The loaded SkipGram model.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

# Sử dụng lớp SkipGramModel
# texts = [
#     # Existing texts from your corpus
#     "Tôi yêu học máy.",
#     "Tôi thích máy học"
#     "Bạn có thích học ngôn ngữ không?",
#     "Học lập trình rất thú vị và bổ ích.",
#     "Tôi thích làm việc với dữ liệu lớn.",
#     "Ngày mai tôi sẽ đi học.",
#     "Giáo dục giúp nâng cao nhận thức và kỹ năng của con người.",
#     "Chúng ta cần đầu tư nhiều hơn vào giáo dục để phát triển xã hội.",
#     "Học từ là quá trình không ngừng nghỉ trong suốt cả cuộc đời.",
#     "Bữa ăn là thời gian tuyệt vời để trao đổi và học hỏi.",
#     "Học cách ăn uống lành mạnh giúp cải thiện sức khỏe.",
#     "Chế độ ăn uống cân bằng đóng vai trò quan trọng trong việc học và làm việc hiệu quả.",
#     "Học nấu ăn là một kỹ năng cần thiết trong đời sống hàng ngày.",
#     "Tôi thích học lập trình và yêu thích công nghệ.",
#     "Chúng ta đều yêu thích làm việc cùng nhau trong nhóm.",
#     "Tôi thích đọc sách về khoa học và yêu thích nghiên cứu.",
#     "Anh ấy yêu thích lập trình và tôi thích việc giảng dạy cho anh ấy.",
#     "Tôi thích đi bộ ngoài trời, nhưng lại ghét phải dậy sớm.",
#     "Chúng ta thích khám phá các nền văn hóa mới, nhưng ghét sự đông đúc.",
#     "Tôi thích ăn sáng ở quán cà phê, nhưng ghét thức ăn nhanh.",
#     "Nhiều người thích chơi thể thao, nhưng cũng có những người ghét vận động."
#     "Tôi thích học toán vì nó rất thú vị và đầy thử thách.",
#     "Bạn đã từng nghĩ đến việc học một ngôn ngữ lập trình mới chưa?",
#     "Học tiếng Trung rất bổ ích và mở ra nhiều cơ hội việc làm.",
#     "Tôi đang nghiên cứu về các thuật toán học sâu.",
#     "Cuộc sống là một hành trình học hỏi không ngừng.",
#     "Hôm nay tôi sẽ thử nấu một món ăn mới.",
#     "Bạn có muốn tham gia câu lạc bộ lập trình cùng tôi không?",
#     "Mỗi ngày, tôi đều cố gắng học thêm một điều mới.",
#     "Đọc sách vào buổi tối giúp tôi thư giãn và học được nhiều kiến thức.",
#     "Tôi rất đam mê với việc giải các bài toán phức tạp.",
#     "Bạn có muốn học thêm về trí tuệ nhân tạo không?",
#     "Đi du lịch đến các quốc gia khác giúp tôi hiểu thêm về văn hóa và con người.",
#     "Tôi thích học những thứ mới, nhưng cũng cảm thấy khó khăn khi bắt đầu.",
#     "Chúng ta hãy cùng nhau làm bài tập này, tôi nghĩ nó sẽ rất thú vị.",
#     "Bạn có lời khuyên nào cho việc học hiệu quả hơn không?",
#     "Hôm qua tôi tham gia một buổi hội thảo về lập trình web.",
#     "Mình thích chia sẻ kiến thức với bạn bè và học từ họ.",
#     "Tôi đang cố gắng hoàn thiện dự án lập trình của mình.",
#     "Mỗi ngày, tôi đều dành thời gian để học thêm một kỹ năng mới.",
#     "Học ngoại ngữ giúp tôi kết nối với nhiều người trên thế giới.",
#     "Bạn có thích nghiên cứu về công nghệ blockchain không?",
#     "Tôi cảm thấy thật hứng thú khi học về trí tuệ nhân tạo và máy học.",
#     "Sáng nay, tôi đã đọc một bài báo thú vị về học sâu.",
#     "Tôi thích tham gia các khóa học trực tuyến vì tính linh hoạt của nó.",
#     "Chúng ta cần làm việc chăm chỉ và không ngừng học hỏi.",
#     "Bạn nghĩ sao về việc học qua các ứng dụng di động?",
#     "Hôm nay tôi sẽ học cách làm một món tráng miệng ngon.",
#     "Cảm giác thật tuyệt vời khi hoàn thành một bài toán khó.",
#     "Bạn có muốn thử học Python cùng tôi không?",
#     "Học cách giải quyết vấn đề là một kỹ năng quan trọng trong lập trình.",
#     "Chúng tôi đã lên kế hoạch cho một khóa học mới về phát triển phần mềm.",
#     "Bạn có bao giờ cảm thấy học tập là một thử thách lớn không?",
#     "Tôi thích nghe podcast về lập trình và công nghệ.",
#     "Cuối tuần này, tôi sẽ thử tham gia một buổi học yoga.",
#     "Chúng ta nên dành thời gian để tự học mỗi ngày.",
#     "Bạn đã từng thử lập trình một ứng dụng di động chưa?",
#     "Tôi thích thảo luận về các xu hướng mới trong công nghệ.",
#     "Mỗi ngày, tôi đều cảm thấy mình tiến bộ hơn một chút nhờ học tập.",
#     "Học cách quản lý thời gian là điều rất quan trọng trong cuộc sống.",
#     "Học là con đường dẫn đến thành công.",
#     "Tôi đang học cách phân tích dữ liệu bằng Python.",
#     "Bạn có thể học cách lập trình trong vòng một tháng không?",
#     "Học cách giao tiếp hiệu quả là một kỹ năng cần thiết.",
#     "Tôi muốn học thêm về các mô hình học sâu.",
#     "Chúng ta hãy cùng nhau học cách giải quyết vấn đề này.",
#     "Học làm bánh là sở thích mới của tôi.",
#     "Học ngôn ngữ giúp tôi mở rộng tầm nhìn.",
#     "Tôi nghĩ học từ sách vở và thực tế đều rất quan trọng.",
#     "Bạn có muốn học chung với tôi vào tối nay không?",
#     "Tôi thích học cách nấu những món ăn mới.",
#     "Bạn có thích xem phim tài liệu không?",
#     "Tôi thích đi bộ vào buổi sáng để thư giãn.",
#     "Chúng ta đều thích chia sẻ kiến thức với nhau.",
#     "Thích chơi thể thao là cách để giữ sức khỏe tốt.",
#     "Bạn có thích học ngôn ngữ mới không?",
#     "Tôi rất thích tham gia các buổi hội thảo về công nghệ.",
#     "Thích nghe nhạc cổ điển giúp tôi tập trung hơn.",
#     "Tôi thích đọc sách vào buổi tối.",
#     "Bạn thích học theo nhóm hay học một mình hơn?",
#     "Tôi yêu học máy vì nó giúp tôi hiểu rõ hơn về công nghệ.",
#     "Bạn có yêu thích việc khám phá những điều mới không?",
#     "Yêu việc học giúp tôi luôn tiến bộ mỗi ngày.",
#     "Tôi yêu cách mà toán học giải thích mọi thứ trong cuộc sống.",
#     "Chúng ta cần yêu thích những gì mình đang làm.",
#     "Bạn có yêu những cuốn sách về triết học không?",
#     "Yêu công việc hiện tại là chìa khóa để thành công.",
#     "Tôi yêu cảm giác khi hoàn thành một bài tập khó.",
#     "Bạn có yêu các thử thách trong học tập không?",
#     "Yêu thương và học hỏi luôn đi đôi với nhau.",
#     "Tôi thích học về các ngôn ngữ lập trình mới.",
#     "Bạn có yêu thích môn toán không?",
#     "Học cách thích nghi với thay đổi là kỹ năng quan trọng.",
#     "Yêu việc học là cách để luôn tiến lên phía trước.",
#     "Tôi thích tham gia các khóa học trực tuyến vì nó tiện lợi.",
#     "Học những điều mình yêu thích làm cho cuộc sống thú vị hơn.",
#     "Bạn có muốn học cùng tôi không? Tôi rất thích điều đó.",
#     "Tôi yêu thích những buổi tối yên tĩnh để đọc sách và học hỏi.",
#     "Học cách yêu bản thân là bước đầu để hạnh phúc.",
#     "Bạn thích học từ sách hay từ video hướng dẫn hơn?",
#     "Học cách giải quyết vấn đề phức tạp là một kỹ năng cần thiết.",
#     "Chúng ta có thể học bất cứ điều gì nếu có quyết tâm.",
#     "Tôi muốn học thêm về cách thiết kế giao diện người dùng.",
#     "Bạn đã bao giờ thử học qua các trò chơi giáo dục chưa?",
#     "Học lịch sử giúp tôi hiểu thêm về thế giới.",
#     "Tôi cảm thấy việc học ngoại ngữ giúp mở rộng tầm nhìn của mình.",
#     "Chúng ta cần học cách làm việc nhóm hiệu quả hơn.",
#     "Học cách đối mặt với thất bại là một phần quan trọng của cuộc sống.",
#     "Tôi đang học cách làm việc với các công cụ AI.",
#     "Học về văn hóa khác là một trải nghiệm thú vị.",
#     "Bạn đã học được điều gì mới hôm nay?",
#     "Tôi nghĩ học online có thể rất hiệu quả nếu biết cách sử dụng.",
#     "Học cách lắng nghe người khác là kỹ năng quan trọng.",
#     "Bạn có muốn học cách viết code hiệu quả hơn không?",
#     "Học từ kinh nghiệm thực tế giúp tôi nhớ lâu hơn.",
#     "Tôi đang học làm sao để cân bằng công việc và cuộc sống.",
#     "Học cách phân tích dữ liệu đã thay đổi cách tôi nhìn nhận thế giới.",
#     "Chúng ta nên học cách quản lý thời gian thật tốt.",
#     "Học toán giúp tôi rèn luyện tư duy logic.",
#     "Tôi rất thích tham gia các câu lạc bộ học thuật.",
#     "Bạn có thích khám phá những công nghệ mới không?",
#     "Tôi thích học các chủ đề liên quan đến khoa học và vũ trụ.",
#     "Bạn có thích học theo nhóm không hay bạn thích học một mình?",
#     "Tôi thích viết blog để chia sẻ những gì tôi đã học.",
#     "Chúng ta đều thích những khoảnh khắc yên bình sau một ngày làm việc căng thẳng.",
#     "Tôi thích tìm hiểu về các hệ thống sinh thái tự nhiên.",
#     "Bạn có thích học về lịch sử Việt Nam không?",
#     "Tôi thích chơi các trò chơi giáo dục với bạn bè.",
#     "Thích làm việc trong môi trường sáng tạo giúp tôi có động lực.",
#     "Tôi rất thích tham gia các câu lạc bộ học thuật.",
#     "Bạn có thích khám phá những công nghệ mới không?",
#     "Tôi thích học các chủ đề liên quan đến khoa học và vũ trụ.",
#     "Bạn có thích học theo nhóm không hay bạn thích học một mình?",
#     "Tôi thích viết blog để chia sẻ những gì tôi đã học.",
#     "Chúng ta đều thích những khoảnh khắc yên bình sau một ngày làm việc căng thẳng.",
#     "Tôi thích tìm hiểu về các hệ thống sinh thái tự nhiên.",
#     "Bạn có thích học về lịch sử Việt Nam không?",
#     "Tôi thích chơi các trò chơi giáo dục với bạn bè.",
#     "Thích làm việc trong môi trường sáng tạo giúp tôi có động lực.",
#     "Tôi rất thích tham gia các câu lạc bộ học thuật.",
#     "Bạn có thích khám phá những công nghệ mới không?",
#     "Tôi thích học các chủ đề liên quan đến khoa học và vũ trụ.",
#     "Bạn có thích học theo nhóm không hay bạn thích học một mình?",
#     "Tôi thích viết blog để chia sẻ những gì tôi đã học.",
#     "Chúng ta đều thích những khoảnh khắc yên bình sau một ngày làm việc căng thẳng.",
#     "Tôi thích tìm hiểu về các hệ thống sinh thái tự nhiên.",
#     "Bạn có thích học về lịch sử Việt Nam không?",
#     "Tôi thích chơi các trò chơi giáo dục với bạn bè.",
#     "Thích làm việc trong môi trường sáng tạo giúp tôi có động lực.",
#     "Tôi yêu cách mà công nghệ đang thay đổi thế giới.",
#     "Bạn có yêu thích việc khám phá những ý tưởng mới không?",
#     "Tôi yêu đọc sách vì nó giúp tôi mở rộng kiến thức.",
#     "Chúng ta nên yêu công việc mình đang làm để cảm thấy hạnh phúc.",
#     "Tôi yêu cảm giác khi giải được một bài toán khó.",
#     "Bạn có yêu thích việc học qua các khóa học trực tuyến không?",
#     "Tôi yêu những khoảnh khắc thư giãn sau khi hoàn thành công việc.",
#     "Học cách yêu thương bản thân là điều quan trọng nhất.",
#     "Tôi yêu việc dạy học vì tôi thích nhìn thấy người khác tiến bộ.",
#     "Bạn có yêu cảm giác hoàn thành mục tiêu của mình không?"
# ]

texts = []
with open('data_vietnamese.txt', 'r',encoding='utf-8') as f:
    for line in f:
        texts.append(line)





# Huấn luyện mô hình
skipgram_model = SkipGramModel(texts, embedding_dim=100, epochs=200, learning_rate=0.01)
skipgram_model.train()
save_model_pickle(skipgram_model, 'skipgram_model')

# word_pairs = [("học", "giáo_dục"), ("lập_trình", "dữ_liệu"), ("thích", "yêu"),("thích","ghét"),("học", "mới")]
word_pairs_test = [("học","giáo_dục"),("học", "ăn_uống")]
# print(skipgram_model.learning_rate)
skipgram_model.evaluate_similarity(word_pairs_test)
# # Tiếp tục huấn luyện mô hình sau khi đã tải
# # skipgram_model.train()


vectors = np.array(list(skipgram_model.W1)[1:])
word_list = list(skipgram_model.vocab.keys())[1:]
cosine_sim_matrix = cosine_similarity(vectors)

# Vẽ heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cosine_sim_matrix, annot=True, cmap="coolwarm", xticklabels=word_list, yticklabels=word_list)
plt.title("Cosine Similarity Heatmap")
plt.show()
