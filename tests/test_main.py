import unittest
from localLLM import LocalLlamaLLM

from main import main

class GraphRagExample(unittest.TestCase):
    
    def test_main(self):
        """Run graph RAG main method."""

        user_question = "How many actors are in the movie speed racer?"

        main(user_question, True)

        # Placeholder test as method used for debug purposes...
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
