
import unittest
from agent_system import AgentSystem
import os

class TestAgentCapabilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n🚀 Initializing Agent for Benchmarking (Model: qwen2.5:14b)...")
        cls.agent = AgentSystem(model_name="qwen2.5:14b")

    def test_01_math_reasoning(self):
        print("\n🧪 Test 1: Math & Reasoning (Fibonacci)")
        query = "Calculate the 10th Fibonacci number using Python."
        response = self.agent.run(query)
        print(f"   Agent Output: {response}")
        self.assertIn("55", response) # 0,1,1,2,3,5,8,13,21,34,55

    def test_02_file_write_read(self):
        print("\n🧪 Test 2: File Operations")
        filename = "benchmark_test.txt"
        content = "Benchmark Verified"
        
        # Write
        self.agent.run(f"Create a file named '{filename}' with content '{content}'.")
        self.assertTrue(os.path.exists(filename))
        
        # Read
        response = self.agent.run(f"Read the content of '{filename}'.")
        print(f"   Agent Output: {response}")
        self.assertIn(content, response)

    def test_03_web_search(self):
        print("\n🧪 Test 3: Web Search")
        query = "Who won the Super Bowl in 2024? State the team name."
        response = self.agent.run(query)
        print(f"   Agent Output: {response}")
        self.assertTrue("Chiefs" in response or "Kansas City" in response)

    def test_04_memory_storage(self):
        print("\n🧪 Test 4: Memory Storage & Recall")
        secret = "Project Apollo"
        self.agent.run(f"Remember that the secret code is '{secret}'.")
        
        response = self.agent.run("What is the secret code I told you to remember?")
        print(f"   Agent Output: {response}")
        self.assertIn(secret, response)

    def test_05_browser_tool(self):
        print("\n🧪 Test 5: Browser Tool (Simulation)")
        # Note: We won't assert exact html content as it varies, just that it didn't fail/refuse.
        query = "Visit example.com and tell me the page title."
        response = self.agent.run(query)
        print(f"   Agent Output: {response}")
        self.assertIn("Example Domain", response)

if __name__ == '__main__':
    unittest.main()
