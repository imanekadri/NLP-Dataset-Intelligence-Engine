
from bertopic import BERTopic

class TopicModeler:

        def __init__(self):
            self.model = BERTopic()

        def get_topic(self, text):
            topics, _ = self.model.fit_transform([text])
            return topics[0]