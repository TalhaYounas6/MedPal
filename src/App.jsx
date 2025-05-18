import { useState } from "react";

export default function App() {
  const [question, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false); 

  const askQuestion = async () => {
    setLoading(true); 
    setAnswer(""); 
    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question }),
      });
      const data = await response.json();
      setAnswer(data.answer);
    } catch (err) {
      setAnswer("An error occurred. Please try again.");
    } finally {
      setLoading(false); 
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-700 to-black flex flex-col items-center justify-center p-4 text-white">
      <div className="w-full max-w-xl bg-gray-800 backdrop-blur-sm p-8 rounded-2xl shadow-2xl border border-gray-700">
        <h1 className="text-4xl font-bold mb-6 text-center bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
          MedPal
        </h1>

        <textarea
          value={question}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask your medical question..."
          className="w-full p-4 rounded-xl bg-gray-700/50 border border-gray-600 shadow-inner text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500/50 transition duration-300"
          rows={4}
        />

        <div className="flex justify-center mt-4">
          <button
            onClick={askQuestion}
            className="px-8 py-3 rounded-xl font-medium bg-gradient-to-r from-blue-500 to-purple-600 hover:from-black hover:to-black shadow-lg transition duration-800 disabled:opacity-50"
            disabled={loading}
          >
            {loading ? "Loading..." : "Ask"}
          </button>
        </div>

        {answer && !loading && (
          <div className="mt-8 bg-gray-700/40 p-6 rounded-xl shadow-lg border border-gray-600/50">
            <div className="font-semibold text-purple-300 mb-2">Answer:</div>
            <div className="text-gray-200">{answer}</div>
          </div>
        )}
      </div>
    </div>
  );
}
