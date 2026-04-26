from bertopic import BERTopic

class TopicModeler:
    def __init__(self):
        self.model = BERTopic(verbose=False)

    def extract_topics(self, texts):
        if len(texts) < 10:
            # Not enough data for BERTopic
            return ["insufficient_data"]

        try:
            topics, _ = self.model.fit_transform(texts)
            topic_info = self.model.get_topic_info()
            top_topics = topic_info.head(5)["Name"].tolist()
            return top_topics
        except Exception as e:
            return ["topic_extraction_failed"]