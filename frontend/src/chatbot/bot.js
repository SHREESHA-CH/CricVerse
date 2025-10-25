import React, { useState, useEffect, useRef } from 'react';
import './bot.css';

const Chatbot = () => {
  const [message, setMessage] = useState('');
  const [chat, setChat] = useState([]);
  const [suggestionsVisible, setSuggestionsVisible] = useState(true);
  const chatboxRef = useRef(null);

  const handleInputChange = (event) => {
    setMessage(event.target.value);
  };

  const handleSend = () => {
    if (message.trim()) {
      setChat([...chat, { text: message, sender: 'user' }]);
      setMessage('');
      setSuggestionsVisible(false);

      fetchResponse(message);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleSend();
    }
  };
  //new
  const fetchResponse = async (userMessage) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: userMessage,
          role: 'user',
        }),
      });
      const data = await response.json();
      setChat((prevChat) => [
        ...prevChat,
        { text: data.text, sender: 'bot' },
      ]);
    } catch (error) {
      console.error('Error fetching response:', error);
      setChat((prevChat) => [
        ...prevChat,
        { text: 'Sorry, something went wrong.', sender: 'bot' },
      ]);
    }
  };
  // const fetchResponse = async (userMessage) => {
  //   setTimeout(() => {
  //     setChat((prevChat) => [
  //       ...prevChat,
  //       { text: getBotResponse(userMessage), sender: 'bot' }
  //     ]);
  //   }, 500);
  // };

  const getBotResponse = (userMessage) => {
    if (userMessage.toLowerCase().includes('score')) {
      return 'Here are the latest cricket scores...';
    } else if (userMessage.toLowerCase().includes('schedule')) {
      return 'The next match is on...';
    } else {
      return 'I am a cricket chatbot. How can I help you?';
    }
  };

  useEffect(() => {
    if (chatboxRef.current) {
      chatboxRef.current.scrollTop = chatboxRef.current.scrollHeight;
    }
  }, [chat]);

  return (
    <div>
      <div className="chatbot-container"><h1>CricVerse</h1>
        <div className="chatbox" ref={chatboxRef}>
          {/* {suggestionsVisible && (
            <div className="Suggestions">
              <p className="Sugg_1" onClick={() => setMessage('Who is the father of pull shot?')}>Who is the father of pull shot?</p>
              <p className="Sugg_2" onClick={() => setMessage('Who is Thala?')}>Who is Thala?</p>
              <p className="Sugg_3" onClick={() => setMessage('What\'s the score today?')}>What's the score today?</p>
            </div>
          )} */}
          {chat.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>

              {msg.sender === 'bot' && <span className="bot-symbol">🤖</span>}
              {msg.text}
            </div>
          ))}
        </div>
        <div className="input-box rounded-3">
          <input
            type="text"
            placeholder="Enter your message"
            value={message}
            onChange={handleInputChange}
            onKeyDown={handleKeyPress}
          />
          <button onClick={handleSend}>Send</button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;