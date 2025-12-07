import { useState, useEffect } from 'react';
import Head from 'next/head';

interface Email {
    id: string;
    sender: string;
    recipients: string[];
    subject: string;
    body: string;
    timestamp: string;
    encrypted: boolean;
    read: boolean;
}

export default function Home() {
    const [userEmail, setUserEmail] = useState('user@senasaitech.app');
    const [inbox, setInbox] = useState<Email[]>([]);
    const [selectedEmail, setSelectedEmail] = useState<Email | null>(null);
    const [composing, setComposing] = useState(false);
    const [stats, setStats] = useState<any>(null);

    // Compose form state
    const [to, setTo] = useState('');
    const [subject, setSubject] = useState('');
    const [body, setBody] = useState('');
    const [sending, setSending] = useState(false);

    // Backend URL - update this for production
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

    useEffect(() => {
        loadInbox();
        loadStats();

        // Real-time updates via WebSocket
        const ws = new WebSocket(`${API_URL.replace('http', 'ws')}/ws/inbox/${userEmail}`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'new_emails') {
                loadInbox();
            }
        };

        return () => ws.close();
    }, [userEmail]);

    const loadInbox = async () => {
        try {
            const response = await fetch(`${API_URL}/api/email/inbox/${userEmail}`);
            const data = await response.json();
            setInbox(data.emails.reverse()); // Show newest first
        } catch (error) {
            console.error('Failed to load inbox:', error);
        }
    };

    const loadStats = async () => {
        try {
            const response = await fetch(`${API_URL}/api/stats`);
            const data = await response.json();
            setStats(data);
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    };

    const sendEmail = async (e: React.FormEvent) => {
        e.preventDefault();
        setSending(true);

        try {
            const response = await fetch(`${API_URL}/api/email/send`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sender: userEmail,
                    recipients: to.split(',').map(e => e.trim()),
                    subject,
                    body,
                    encrypt: true
                })
            });

            const result = await response.json();

            if (result.success) {
                alert(`Email sent in ${result.delivery_time_ms.toFixed(2)}ms! 🚀`);
                setComposing(false);
                setTo('');
                setSubject('');
                setBody('');
            }
        } catch (error) {
            alert('Failed to send email: ' + error);
        } finally {
            setSending(false);
        }
    };

    const markAsRead = async (emailId: string) => {
        try {
            await fetch(`${API_URL}/api/email/mark-read/${emailId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_email: userEmail })
            });
            loadInbox();
        } catch (error) {
            console.error('Failed to mark as read:', error);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-black text-white">
            <Head>
                <title>Queztl Email - Faster than ProtonMail</title>
                <meta name="description" content="Lightning-fast, secure email powered by Queztl" />
            </Head>

            {/* Header */}
            <header className="bg-black/50 backdrop-blur-lg border-b border-purple-500/30 p-4">
                <div className="max-w-7xl mx-auto flex justify-between items-center">
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                        ⚡ Queztl Email
                    </h1>
                    <div className="flex gap-4 items-center">
                        {stats && (
                            <div className="text-sm text-gray-300">
                                <span className="text-green-400">{stats.performance.avg_delivery_ms}ms</span> avg delivery |
                                <span className="text-blue-400"> {stats.performance.throughput_rps}</span> req/sec
                            </div>
                        )}
                        <button
                            onClick={() => setComposing(true)}
                            className="bg-gradient-to-r from-purple-500 to-pink-500 px-6 py-2 rounded-lg font-semibold hover:scale-105 transition"
                        >
                            ✉️ Compose
                        </button>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto p-6">
                {/* Stats Banner */}
                {stats && (
                    <div className="bg-black/40 backdrop-blur-lg border border-purple-500/30 rounded-xl p-6 mb-6">
                        <div className="grid grid-cols-4 gap-6 text-center">
                            <div>
                                <div className="text-3xl font-bold text-purple-400">{stats.performance.avg_delivery_ms}ms</div>
                                <div className="text-sm text-gray-400">Avg Delivery</div>
                            </div>
                            <div>
                                <div className="text-3xl font-bold text-blue-400">{stats.performance.throughput_rps}</div>
                                <div className="text-sm text-gray-400">Requests/sec</div>
                            </div>
                            <div>
                                <div className="text-3xl font-bold text-green-400">{stats.total_emails}</div>
                                <div className="text-sm text-gray-400">Total Emails</div>
                            </div>
                            <div>
                                <div className="text-3xl font-bold text-pink-400">{stats.performance.uptime_percent}%</div>
                                <div className="text-sm text-gray-400">Uptime</div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Compose Modal */}
                {composing && (
                    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
                        <div className="bg-gray-900 border border-purple-500/30 rounded-xl p-8 max-w-2xl w-full">
                            <h2 className="text-2xl font-bold mb-6">Compose Email</h2>
                            <form onSubmit={sendEmail}>
                                <input
                                    type="text"
                                    placeholder="To: (comma-separated for multiple)"
                                    value={to}
                                    onChange={(e) => setTo(e.target.value)}
                                    className="w-full bg-black/50 border border-purple-500/30 rounded-lg p-3 mb-4"
                                    required
                                />
                                <input
                                    type="text"
                                    placeholder="Subject"
                                    value={subject}
                                    onChange={(e) => setSubject(e.target.value)}
                                    className="w-full bg-black/50 border border-purple-500/30 rounded-lg p-3 mb-4"
                                    required
                                />
                                <textarea
                                    placeholder="Message"
                                    value={body}
                                    onChange={(e) => setBody(e.target.value)}
                                    className="w-full bg-black/50 border border-purple-500/30 rounded-lg p-3 mb-4 h-64"
                                    required
                                />
                                <div className="flex gap-4">
                                    <button
                                        type="submit"
                                        disabled={sending}
                                        className="flex-1 bg-gradient-to-r from-purple-500 to-pink-500 px-6 py-3 rounded-lg font-semibold hover:scale-105 transition disabled:opacity-50"
                                    >
                                        {sending ? 'Sending...' : '🚀 Send Email'}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setComposing(false)}
                                        className="px-6 py-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition"
                                    >
                                        Cancel
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}

                {/* Email List and Reader */}
                <div className="grid grid-cols-3 gap-6">
                    {/* Inbox List */}
                    <div className="col-span-1 bg-black/40 backdrop-blur-lg border border-purple-500/30 rounded-xl p-4 max-h-[600px] overflow-y-auto">
                        <h2 className="text-xl font-bold mb-4">Inbox ({inbox.length})</h2>
                        {inbox.length === 0 ? (
                            <p className="text-gray-400 text-center py-8">No emails yet</p>
                        ) : (
                            inbox.map((email) => (
                                <div
                                    key={email.id}
                                    onClick={() => {
                                        setSelectedEmail(email);
                                        if (!email.read) markAsRead(email.id);
                                    }}
                                    className={`p-4 mb-2 rounded-lg cursor-pointer transition ${selectedEmail?.id === email.id
                                            ? 'bg-purple-500/30 border border-purple-400'
                                            : 'bg-gray-800/50 hover:bg-gray-700/50'
                                        } ${!email.read ? 'font-bold' : ''}`}
                                >
                                    <div className="text-sm text-purple-400">{email.sender}</div>
                                    <div className="font-semibold truncate">{email.subject}</div>
                                    <div className="text-xs text-gray-400">{new Date(email.timestamp).toLocaleString()}</div>
                                </div>
                            ))
                        )}
                    </div>

                    {/* Email Reader */}
                    <div className="col-span-2 bg-black/40 backdrop-blur-lg border border-purple-500/30 rounded-xl p-6">
                        {selectedEmail ? (
                            <>
                                <div className="mb-6">
                                    <h2 className="text-2xl font-bold mb-2">{selectedEmail.subject}</h2>
                                    <div className="text-sm text-gray-400">
                                        From: <span className="text-purple-400">{selectedEmail.sender}</span>
                                    </div>
                                    <div className="text-sm text-gray-400">
                                        To: {selectedEmail.recipients.join(', ')}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-2">
                                        {new Date(selectedEmail.timestamp).toLocaleString()}
                                        {selectedEmail.encrypted && <span className="ml-2 text-green-400">🔒 Encrypted</span>}
                                    </div>
                                </div>
                                <div className="prose prose-invert max-w-none">
                                    <p className="whitespace-pre-wrap">{selectedEmail.body}</p>
                                </div>
                            </>
                        ) : (
                            <div className="flex items-center justify-center h-full text-gray-400">
                                Select an email to read
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
