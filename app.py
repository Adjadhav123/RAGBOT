import os
import uuid
import time as time_module
import platform
import re
import signal
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_document_text(text):
    """
    Clean and preprocess document text for better embedding and retrieval.
    """
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove page numbers and headers/footers patterns
    text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
    text = re.sub(r'Page \d+ of \d+', '', text)  # Remove "Page X of Y"
    
    # Fix common PDF extraction issues
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Add space after punctuation
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{3,}', '...', text)  # Replace multiple dots with ellipsis
    text = re.sub(r'[-]{3,}', '---', text)  # Replace multiple dashes
    
    # Clean up and return
    return text.strip()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
CORS(app)  # Enable CORS for all routes

# Load environment variables
load_dotenv()

# Set up embeddings
try:
    embeddings = OllamaEmbeddings(model="all-minilm")
except Exception as e:
    logger.error(f"Failed to initialize Ollama embeddings: {e}")
    raise ValueError("Could not initialize embeddings. Ensure Ollama is running with all-minilm model.")

# In-memory store for chat history
store = {}

# Ensure uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Groq API
groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY not found in environment variables")

try:
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,  # Slightly higher for more natural, educational responses
        max_tokens=2000,  # Increased for more comprehensive explanations
        streaming=False
    )
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {e}")
    raise ValueError("Could not initialize Groq LLM. Check API key and network.")

# Global variable for RAG chain
conversational_rag_chain = None
document_stats = {"files_processed": 0, "total_chunks": 0, "last_processed": None}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy", "service": "ragbot"}), 200

@app.route('/')
def index():
    logger.info("Serving index.html")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global conversational_rag_chain
    logger.info("Received upload request")
    start_time = time_module.time()

    if 'files' not in request.files:
        logger.warning("No files uploaded in request")
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('files')
    documents = []
    file_paths = []
    
    # Validate file sizes and types
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
    
    for file in files:
        if file and file.filename.endswith('.pdf'):
            # Check file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size > MAX_FILE_SIZE:
                logger.warning(f"File {file.filename} exceeds size limit: {file_size} bytes")
                return jsonify({"error": f"File {file.filename} exceeds 50MB limit"}), 400
            
            filename = str(uuid.uuid4()) + '.pdf'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
            logger.info(f"Saved file {file.filename} as {filename}")
            try:
                logger.info(f"Starting extraction for {file.filename}")
                
                # Create loader with error handling for corrupted PDFs
                loader = PyPDFLoader(file_path)
                
                # Add timeout protection for large PDF processing
                import threading
                import queue
                
                def load_pdf_with_timeout(loader, timeout=30):
                    """Load PDF with timeout protection"""
                    result_queue = queue.Queue()
                    exception_queue = queue.Queue()
                    
                    def load_worker():
                        try:
                            docs = loader.load()
                            result_queue.put(docs)
                        except Exception as e:
                            exception_queue.put(e)
                    
                    thread = threading.Thread(target=load_worker)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout)
                    
                    if thread.is_alive():
                        # Thread is still running, timeout occurred
                        raise TimeoutError("PDF processing timeout")
                    
                    if not exception_queue.empty():
                        raise exception_queue.get()
                    
                    if not result_queue.empty():
                        return result_queue.get()
                    
                    raise Exception("Unknown error during PDF processing")
                
                try:
                    docs = load_pdf_with_timeout(loader, timeout=30)
                    logger.info(f"Loaded {len(docs)} pages from {file.filename}")
                except TimeoutError:
                    logger.error(f"Timeout processing PDF {file.filename}")
                    return jsonify({"error": f"PDF {file.filename} is too complex to process. Please try a simpler PDF."}), 400
                except Exception as pdf_error:
                    logger.error(f"PDF parsing error for {file.filename}: {pdf_error}")
                    return jsonify({"error": f"Unable to read PDF {file.filename}. File may be corrupted or password protected."}), 400
                
                if not docs:
                    logger.warning(f"No pages found in PDF {file.filename}, skipping.")
                    continue
                
                # Remove the sleep to reduce processing time in production
                # time_module.sleep(2)  # Removed for production efficiency
                
                # Limit processing to smaller chunks for memory efficiency
                max_pages = min(50, len(docs))  # Reduced from 100 to 50 pages max
                processed_docs = []
                
                for i, doc in enumerate(docs[:max_pages]):
                    try:
                        # Process each page with error handling
                        if len(doc.page_content.strip()) > 20:  # Only process pages with content
                            doc.page_content = preprocess_document_text(doc.page_content)
                            doc.metadata['source_file'] = file.filename
                            doc.metadata['file_type'] = 'PDF'
                            doc.metadata['page_number'] = i + 1
                            processed_docs.append(doc)
                    except Exception as page_error:
                        logger.warning(f"Error processing page {i+1} of {file.filename}: {page_error}")
                        continue
                
                documents.extend(processed_docs)
                logger.info(f"Successfully processed {len(processed_docs)} pages from {file.filename}")
                
            except Exception as e:
                logger.error(f"Failed to process PDF {file.filename}: {e}")
                # Continue with other files instead of failing completely
                continue
    
    if not documents:
        logger.warning("No valid PDF files uploaded")
        return jsonify({"error": "No valid PDF files uploaded"}), 400
    
    try:
        # Split documents with memory-efficient chunking strategy
        logger.info(f"Splitting {len(documents)} documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Reduced chunk size for better memory usage
            chunk_overlap=150,  # Reduced overlap for memory efficiency
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for meaningful chunks
        )
        
        # Process documents in batches to avoid memory issues
        splits = []
        batch_size = 10  # Process 10 documents at a time
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_splits = text_splitter.split_documents(batch)
            splits.extend(batch_splits)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        # Filter out very short chunks and limit total chunks for memory
        splits = [doc for doc in splits if len(doc.page_content.strip()) > 30]
        max_chunks = 500  # Limit total chunks to avoid memory issues
        if len(splits) > max_chunks:
            splits = splits[:max_chunks]
            logger.info(f"Limited chunks to {max_chunks} for memory efficiency")
        
        logger.info(f"Created {len(splits)} text chunks")
        
        # Create FAISS vector store with optimized configuration
        logger.info("Creating FAISS vector store")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        # Configure retriever to get more relevant documents
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 6,  # Retrieve top 6 most relevant chunks
                "fetch_k": 20  # Consider top 20 before filtering to top 6
            }
        )
        
        # Contextualize question prompt - Enhanced
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question that captures the full intent "
            "and context. Consider any pronouns, references, or implicit context from the conversation. "
            "Make the question specific and detailed enough to retrieve the most relevant information "
            "from the document. If the question is already standalone, return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        # Create history-aware retriever
        logger.info("Creating history-aware retriever")
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # Question-answering prompt - Enhanced for student-friendly responses
        system_prompt = (
            "You are a helpful educational assistant designed to help students understand complex topics. "
            "Your goal is to provide clear, well-structured, and easy-to-understand answers based on the document content.\n\n"
            
            "FORMATTING GUIDELINES:\n"
            "• Use clear headings and bullet points to organize information\n"
            "• Break down complex concepts into simple, digestible parts\n"
            "• Use numbered lists for step-by-step explanations\n"
            "• Include examples when helpful for understanding\n"
            "• Use bold text for **key terms** and important concepts\n"
            "• Separate different topics with clear sections\n\n"
            
            "RESPONSE STRUCTURE:\n"
            "1. **Main Answer**: Start with a direct answer to the question\n"
            "2. **Detailed Explanation**: Provide comprehensive details in an organized manner\n"
            "3. **Key Points**: Highlight the most important takeaways\n"
            "4. **Examples**: Include relevant examples from the document when available\n"
            "5. **Summary**: End with a brief summary if the answer is long\n\n"
            
            "LANGUAGE GUIDELINES:\n"
            "• Use simple, clear language that students can easily understand\n"
            "• Explain technical terms when you first use them\n"
            "• Use active voice and conversational tone\n"
            "• Avoid jargon unless necessary (and explain it if used)\n"
            "• Make connections between different concepts when relevant\n\n"
            
            "SPECIAL INSTRUCTIONS:\n"
            "• If information is not in the context, clearly state: 'This information is not available in the provided documents'\n"
            "• When referencing specific sections, mention the source document if multiple files were uploaded\n"
            "• If a concept has multiple aspects, organize them clearly with subheadings\n"
            "• Always aim to be educational and help the student learn, not just answer\n\n"
            
            "Context from the uploaded document(s):\n{context}\n\n"
            
            "Please provide a well-formatted, student-friendly answer to the following question:"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Create RAG chain
        logger.info("Creating RAG chain")
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Create conversational RAG chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        logger.info("Conversational RAG chain initialized successfully")
        
        # Update document statistics
        document_stats["files_processed"] = len(files)
        document_stats["total_chunks"] = len(splits)
        document_stats["last_processed"] = time_module.strftime("%Y-%m-%d %H:%M:%S")
        
        # Clean up uploaded files
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to delete file {file_path}: {e}")
        
        processing_time = time_module.time() - start_time
        logger.info(f"PDF(s) processed successfully in {processing_time:.2f} seconds")
        return jsonify({"message": "PDF(s) processed successfully"})
    
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global conversational_rag_chain
    logger.info("Received chat request")
    data = request.json
    user_input = data.get('input')
    session_id = data.get('session_id', 'default_session')
    
    if not user_input:
        logger.warning("No input provided in chat request")
        return jsonify({"error": "No input provided"}), 400
    
    if not conversational_rag_chain:
        logger.warning("Chat attempted before PDF upload")
        return jsonify({"error": "Please upload a PDF first"}), 400
    
    try:
        # Add timeout protection for chat responses using threading
        import threading
        import queue
        
        def chat_with_timeout(chain, input_data, config, timeout=45):
            """Process chat with timeout protection"""
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            
            def chat_worker():
                try:
                    response = chain.invoke(input_data, config=config)
                    result_queue.put(response)
                except Exception as e:
                    exception_queue.put(e)
            
            thread = threading.Thread(target=chat_worker)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            
            if thread.is_alive():
                # Thread is still running, timeout occurred
                raise TimeoutError("Chat response timeout")
            
            if not exception_queue.empty():
                raise exception_queue.get()
            
            if not result_queue.empty():
                return result_queue.get()
            
            raise Exception("Unknown error during chat processing")
        
        try:
            response = chat_with_timeout(
                conversational_rag_chain,
                {"input": user_input},
                {"configurable": {"session_id": session_id}},
                timeout=45
            )
        except TimeoutError:
            logger.error("Chat response timeout")
            return jsonify({"error": "Request timeout. Please try a simpler question or check your connection."}), 408
        
        # Extract source information from retrieved documents
        source_info = []
        if 'context' in response:
            for doc in response['context']:
                if hasattr(doc, 'metadata') and 'source_file' in doc.metadata:
                    source_info.append({
                        'file': doc.metadata.get('source_file', 'Unknown'),
                        'page': doc.metadata.get('page', 'Unknown')
                    })
        
        # Remove duplicate sources
        unique_sources = []
        seen = set()
        for source in source_info:
            source_key = f"{source['file']}-{source['page']}"
            if source_key not in seen:
                seen.add(source_key)
                unique_sources.append(source)
        
        session_history = get_session_history(session_id)
        messages = [
            {"type": msg.type, "content": msg.content}
            for msg in session_history.messages
        ]
        
        logger.info(f"Chat response generated for session {session_id}")
        return jsonify({
            "answer": response["answer"],
            "sources": unique_sources[:3],  # Limit to top 3 sources
            "history": messages
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    logger.info("Received clear history request")
    data = request.json
    session_id = data.get('session_id', 'default_session')
    try:
        if session_id in store:
            store[session_id] = ChatMessageHistory()
        logger.info(f"Chat history cleared for session {session_id}")
        return jsonify({"message": "Chat history cleared"})
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/document_stats', methods=['GET'])
def get_document_stats():
    """Get statistics about processed documents"""
    return jsonify(document_stats)

if __name__ == '__main__':
    # Disable reloader on Windows to avoid socket error
    use_reloader = False if platform.system() == "Windows" else True
    app.run(debug=True, host='0.0.0.0', port=3000, use_reloader=use_reloader)