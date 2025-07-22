// Production Chatbot Frontend - app.js
// Enhanced version with modern UI and production features

class ChatbotApp {
    constructor() {
        // Modified to connect to Flask server on port 5000
        this.apiUrl = window.location.origin; // Changed from window.location.origin
        this.isInitialized = false;
        this.currentView = 'home';
        this.chatHistory = [];
        this.knowledgeBaseFiles = [];
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.showHomeView();
        await this.checkHealth();
    }

    setupEventListeners() {
        // Navigation buttons
        document.getElementById('homeBtn')?.addEventListener('click', () => this.showHomeView());
        document.getElementById('chatBtn')?.addEventListener('click', () => this.showChatView());
        document.getElementById('driveKbBtn')?.addEventListener('click', () => this.showKnowledgeBaseView());
        
        // Management buttons
        document.getElementById('reloadBtn')?.addEventListener('click', () => this.reloadChatbot());
        document.getElementById('quitBtn')?.addEventListener('click', () => this.quitServer());
        document.getElementById('initializeBtn')?.addEventListener('click', () => this.initializeChatbot());
        
        // In your setupEventListeners method, update this line:

        // Refresh knowledge base - UPDATED TO INCLUDE SYNC
        document.getElementById('refreshKbBtn')?.addEventListener('click', () => this.refreshKnowledgeBase());

        // Chat functionality
        document.getElementById('sendBtn')?.addEventListener('click', () => this.sendMessage());
        document.getElementById('chatInput')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // File details
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('file-detail-btn')) {
                const filename = e.target.dataset.filename;
                this.showFileDetails(filename);
            }
        });

        // Clear chat
        document.getElementById('clearChatBtn')?.addEventListener('click', () => this.clearChat());
        
        // Refresh knowledge base
        document.getElementById('refreshKbBtn')?.addEventListener('click', () => this.loadKnowledgeBase());
    }

    showHomeView() {
        this.currentView = 'home';
        this.hideAllViews();
        document.getElementById('homeView').style.display = 'block';
        this.updateNavigation();
    }

    showChatView() {
        this.currentView = 'chat';
        this.hideAllViews();
        document.getElementById('chatView').style.display = 'block';
        this.updateNavigation();
        
        // Focus on chat input
        setTimeout(() => {
            document.getElementById('chatInput')?.focus();
        }, 100);
    }

    async showKnowledgeBaseView() {
        this.currentView = 'knowledge-base';
        this.hideAllViews();
        document.getElementById('knowledgeBaseView').style.display = 'block';
        this.updateNavigation();
        
        // Load knowledge base files
        await this.loadKnowledgeBase();
    }

    hideAllViews() {
        const views = ['homeView', 'chatView', 'knowledgeBaseView'];
        views.forEach(view => {
            const element = document.getElementById(view);
            if (element) element.style.display = 'none';
        });
    }

    updateNavigation() {
        // Update active navigation button
        const navButtons = document.querySelectorAll('.nav-btn');
        navButtons.forEach(btn => btn.classList.remove('active'));
        
        const activeBtn = document.getElementById(`${this.currentView === 'knowledge-base' ? 'driveKb' : this.currentView}Btn`);
        if (activeBtn) activeBtn.classList.add('active');
    }

    async checkHealth() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            
            this.isInitialized = data.initialized;
            this.updateStatus(data.status, data.initialized);
            
            if (!this.isInitialized) {
                this.showNotification('⚠️ Chatbot not initialized. Click Initialize to get started.', 'warning');
            }
            
        } catch (error) {
            console.error('Health check failed:', error);
            this.updateStatus('error', false);
            this.showNotification('❌ Cannot connect to chatbot server on port 5000', 'error');
        }
    }

    updateStatus(status, initialized) {
        const statusElement = document.getElementById('status');
        const statusText = document.getElementById('statusText');
        
        if (statusElement && statusText) {
            statusElement.className = `status ${status}`;
            statusText.textContent = initialized ? 'Ready' : 'Not Initialized';
        }
        
        // Update initialization button
        const initBtn = document.getElementById('initializeBtn');
        if (initBtn) {
            initBtn.style.display = initialized ? 'none' : 'block';
        }
    }

    async initializeChatbot() {
        try {
            this.showLoading('Initializing chatbot...');
            
            const response = await fetch(`${this.apiUrl}/initialize`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.isInitialized = true;
                this.updateStatus('healthy', true);
                this.showNotification('✅ Chatbot initialized successfully!', 'success');
            } else {
                this.showNotification('❌ Initialization failed: ' + (data.error || 'Unknown error'), 'error');
            }
            
        } catch (error) {
            console.error('Initialization failed:', error);
            this.showNotification('❌ Initialization failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async reloadChatbot() {
        if (!confirm('Are you sure you want to reload the chatbot? This will restart all connections.')) {
            return;
        }
        
        try {
            this.showLoading('Reloading chatbot...');
            
            const response = await fetch(`${this.apiUrl}/reload`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.isInitialized = true;
                this.updateStatus('healthy', true);
                this.showNotification('✅ Chatbot reloaded successfully!', 'success');
                
                // Refresh knowledge base if we're on that view
                if (this.currentView === 'knowledge-base') {
                    await this.loadKnowledgeBase();
                }
            } else {
                this.showNotification('❌ Reload failed: ' + (data.error || 'Unknown error'), 'error');
            }
            
        } catch (error) {
            console.error('Reload failed:', error);
            this.showNotification('❌ Reload failed: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async quitServer() {
        if (!confirm('Are you sure you want to quit the server? This will shut down the entire application.')) {
            return;
        }
        
        try {
            this.showLoading('Shutting down server...');
            
            const response = await fetch(`${this.apiUrl}/quit`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.showNotification('🔴 Server is shutting down...', 'info');
                
                // Update UI to show shutdown state
                setTimeout(() => {
                    this.updateStatus('error', false);
                    this.showNotification('❌ Server has been shut down', 'error');
                }, 2000);
            }
            
        } catch (error) {
            // This is expected when server shuts down
            console.log('Server shutdown initiated');
            this.showNotification('🔴 Server shutdown initiated', 'info');
            this.updateStatus('error', false);
        } finally {
            this.hideLoading();
        }
    }
    
    async refreshKnowledgeBase() {
    try {
        // Update button text to show it's working
        const refreshBtn = document.getElementById('refreshKbBtn');
        const originalText = refreshBtn ? refreshBtn.innerHTML : '';
        if (refreshBtn) {
            refreshBtn.innerHTML = '🔄 Syncing...';
            refreshBtn.disabled = true;
        }
        
        // Call loadKnowledgeBase with forceSync = true
        await this.loadKnowledgeBase(true);
        
    } catch (error) {
        console.error('Refresh failed:', error);
        this.showNotification('❌ Refresh failed: ' + error.message, 'error');
    } finally {
        // Restore button
        const refreshBtn = document.getElementById('refreshKbBtn');
        if (refreshBtn) {
            refreshBtn.innerHTML = '🔄 Refresh';
            refreshBtn.disabled = false;
        }
    }
}
    async loadKnowledgeBase(forceSync = false) {
    try {
        // If forceSync is true, run sync first
        if (forceSync) {
            this.showLoading('Syncing knowledge base with filesystem...');
            
            try {
                const syncResponse = await fetch(`${this.apiUrl}/sync`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        force_rebuild: false // Set to true if you want to force rebuild
                    })
                });
                
                const syncData = await syncResponse.json();
                
                if (syncData.success) {
                    this.showNotification(`✅ Sync completed! ${syncData.message || 'Database updated'}`, 'success');
                } else {
                    this.showNotification('⚠️ Sync completed with warnings: ' + (syncData.error || 'Unknown error'), 'warning');
                }
            } catch (syncError) {
                console.error('Sync failed:', syncError);
                this.showNotification('❌ Sync failed: ' + syncError.message, 'error');
                // Continue to load files even if sync failed
            }
        }
        
        // Now load the knowledge base files
        this.showLoading('Loading knowledge base files...');
        
        const response = await fetch(`${this.apiUrl}/files`);
        const data = await response.json();
        
        if (data.success) {
            this.knowledgeBaseFiles = data.files || [];
            this.displayKnowledgeBase(data);
            
            if (!forceSync) {
                this.showNotification('✅ Knowledge base loaded successfully!', 'success');
            }
        } else {
            this.showNotification('❌ Failed to load knowledge base: ' + (data.error || 'Unknown error'), 'error');
            this.displayKnowledgeBaseError(data.error || 'Unknown error');
        }
        
    } catch (error) {
        console.error('Failed to load knowledge base:', error);
        this.showNotification('❌ Failed to load knowledge base: ' + error.message, 'error');
        this.displayKnowledgeBaseError(error.message);
    } finally {
        this.hideLoading();
    }
}

    displayKnowledgeBase(data) {
        const container = document.getElementById('knowledgeBaseContent');
        if (!container) return;
        
        const files = data.files || [];
        const summary = data.summary || '';
        
        let html = `
            <div class="kb-header">
                <h3>📚 Knowledge Base Files</h3>
                <div class="kb-actions">
                    <button id="refreshKbBtn" class="btn btn-secondary">
                        🔄 Refresh
                    </button>
                </div>
            </div>
            
            <div class="kb-summary">
                <pre>${summary}</pre>
            </div>
        `;
        
        if (files.length > 0) {
            html += `
                <div class="kb-files">
                    <h4>📋 File Details</h4>
                    <div class="files-grid">
            `;
            
            files.forEach(file => {
                html += `
                    <div class="file-card">
                        <div class="file-header">
                            <span class="file-icon">${this.getFileIcon(file.file_type)}</span>
                            <span class="file-name">${file.filename}</span>
                        </div>
                        <div class="file-details">
                            <div class="file-stat">
                                <span class="label">Type:</span>
                                <span class="value">${file.file_type}</span>
                            </div>
                            <div class="file-stat">
                                <span class="label">Size:</span>
                                <span class="value">${file.file_size_mb}MB</span>
                            </div>
                            <div class="file-stat">
                                <span class="label">Chunks:</span>
                                <span class="value">${file.chunk_count}</span>
                            </div>
                            <div class="file-stat">
                                <span class="label">Indexed:</span>
                                <span class="value">${new Date(file.indexed_at).toLocaleDateString()}</span>
                            </div>
                        </div>
                        <div class="file-actions">
                            <button class="btn btn-sm btn-outline file-detail-btn" data-filename="${file.filename}">
                                🔍 View Details
                            </button>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        } else {
            html += `
                <div class="empty-state">
                    <h4>📁 No files found</h4>
                    <p>The knowledge base appears to be empty. Add some documents to get started.</p>
                </div>
            `;
        }
        
        container.innerHTML = html;
        
        // Re-attach event listeners
        this.reattachKnowledgeBaseEventListeners();
    }

   reattachKnowledgeBaseEventListeners() {
    // Re-attach refresh button listener - NOW WITH SYNC
    const refreshBtn = document.getElementById('refreshKbBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => this.refreshKnowledgeBase());
    }
    
    // Re-attach all file detail button listeners
    const fileDetailBtns = document.querySelectorAll('.file-detail-btn');
    fileDetailBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const filename = e.target.dataset.filename;
            this.showFileDetails(filename);
        });
    });
}

    displayKnowledgeBaseError(error) {
        const container = document.getElementById('knowledgeBaseContent');
        if (!container) return;
        
        container.innerHTML = `
            <div class="error-state">
                <h4>❌ Error Loading Knowledge Base</h4>
                <p>${error}</p>
                <button class="btn btn-primary" onclick="window.chatbot.loadKnowledgeBase()">
                    🔄 Retry
                </button>
            </div>
        `;
    }

    getFileIcon(fileType) {
        const icons = {
            '.pdf': '📄',
            '.docx': '📝',
            '.doc': '📝',
            '.txt': '📄',
            '.csv': '📊',
            '.xlsx': '📊',
            '.pptx': '📊',
            '.py': '🐍',
            '.js': '💛',
            '.html': '🌐',
            '.md': '📖'
        };
        return icons[fileType] || '📄';
    }

    async showFileDetails(filename) {
        try {
            this.showLoading(`Loading details for ${filename}...`);
            
            const response = await fetch(`${this.apiUrl}/files/${encodeURIComponent(filename)}`);
            const data = await response.json();
            
            if (data.success) {
                this.displayFileDetailsModal(data);
            } else {
                this.showNotification('❌ Failed to load file details: ' + (data.error || 'Unknown error'), 'error');
            }
            
        } catch (error) {
            console.error('Failed to load file details:', error);
            this.showNotification('❌ Failed to load file details: ' + error.message, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayFileDetailsModal(data) {
        const modal = document.getElementById('fileModal') || this.createFileModal();
        const content = modal.querySelector('.modal-content');
        
        content.innerHTML = `
            <div class="modal-header">
                <h3>📄 ${data.filename}</h3>
                <button class="modal-close" onclick="this.closest('.modal').style.display='none'">&times;</button>
            </div>
            <div class="modal-body">
                <div class="file-details-grid">
                    <div class="detail-item">
                        <strong>Chunks:</strong> ${data.chunks_count}
                    </div>
                    <div class="detail-item">
                        <strong>Total Content:</strong> ${data.total_content_length.toLocaleString()} characters
                    </div>
                    <div class="detail-item">
                        <strong>Average Chunk Size:</strong> ${data.average_chunk_size.toLocaleString()} characters
                    </div>
                </div>
                
                <h4>📝 Chunk Preview</h4>
                <div class="chunks-list">
                    ${data.chunks.slice(0, 5).map((chunk, i) => `
                        <div class="chunk-item">
                            <div class="chunk-header">
                                <strong>Chunk ${chunk.chunk_id}</strong>
                                <span class="chunk-size">${chunk.content_length} chars</span>
                            </div>
                            <div class="chunk-preview">
                                ${chunk.content_preview}
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                ${data.chunks.length > 5 ? `
                    <p class="chunk-summary">... and ${data.chunks.length - 5} more chunks</p>
                ` : ''}
            </div>
        `;
        
        modal.style.display = 'block';
    }

    createFileModal() {
        const modal = document.createElement('div');
        modal.id = 'fileModal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <!-- Content will be filled dynamically -->
            </div>
        `;
        document.body.appendChild(modal);
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
        
        return modal;
    }

    async sendMessage() {
        const input = document.getElementById('chatInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        if (!this.isInitialized) {
            this.showNotification('⚠️ Please initialize the chatbot first', 'warning');
            return;
        }
        
        // Clear input and add to chat
        input.value = '';
        this.addMessageToChat('user', message);
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message,
                    k: 5
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.addMessageToChat('assistant', data.response, data.metadata);
            } else {
                this.addMessageToChat('assistant', `❌ Error: ${data.error || 'Unknown error'}`, null, true);
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessageToChat('assistant', `❌ Connection error: ${error.message}`, null, true);
        } finally {
            this.hideTypingIndicator();
        }
    }

    addMessageToChat(sender, message, metadata = null, isError = false) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender} ${isError ? 'error' : ''}`;
        
        let html = `
            <div class="message-content">
                <div class="message-text">${this.formatMessage(message)}</div>
        `;
        
        if (metadata) {
            html += `
                <div class="message-metadata">
                    <small>
                        📄 Found ${metadata.chunks_found} relevant chunks
                        ${metadata.search_filter ? ` • Filtered by: ${metadata.search_filter}` : ''}
                        • ${new Date().toLocaleTimeString()}
                    </small>
                </div>
            `;
        } else if (sender === 'user') {
            html += `
                <div class="message-metadata">
                    <small>${new Date().toLocaleTimeString()}</small>
                </div>
            `;
        }
        
        html += `</div>`;
        messageDiv.innerHTML = html;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Store in history
        this.chatHistory.push({ sender, message, metadata, timestamp: new Date() });
    }

    formatMessage(message) {
        return message.replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const indicator = document.createElement('div');
        indicator.id = 'typingIndicator';
        indicator.className = 'message assistant typing';
        indicator.innerHTML = `
            <div class="message-content">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    clearChat() {
        if (!confirm('Are you sure you want to clear the chat history?')) {
            return;
        }
        
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
        
        this.chatHistory = [];
        this.showNotification('🗑️ Chat cleared', 'info');
    }

    showLoading(message = 'Loading...') {
        const loader = document.getElementById('globalLoader') || this.createGlobalLoader();
        const loaderText = loader.querySelector('.loader-text');
        
        if (loaderText) {
            loaderText.textContent = message;
        }
        
        loader.style.display = 'flex';
    }

    hideLoading() {
        const loader = document.getElementById('globalLoader');
        if (loader) {
            loader.style.display = 'none';
        }
    }

    createGlobalLoader() {
        const loader = document.createElement('div');
        loader.id = 'globalLoader';
        loader.className = 'global-loader';
        loader.innerHTML = `
            <div class="loader-content">
                <div class="spinner"></div>
                <div class="loader-text">Loading...</div>
            </div>
        `;
        document.body.appendChild(loader);
        return loader;
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Show with animation
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Auto hide after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new ChatbotApp();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChatbotApp;
}