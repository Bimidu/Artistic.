'use client';

import { useEffect } from 'react';
import Link from 'next/link';

const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function GuidelinePage() {
    useEffect(() => {
        (window as unknown as { __ARTISTIC_API_URL?: string }).__ARTISTIC_API_URL = apiUrl;
    }, []);

    useEffect(() => {
        const script = document.createElement('script');
        script.src = '/guidelines.js';
        script.async = true;
        document.body.appendChild(script);
    }, []);

    return (
        <>
            <header className="bg-lime-950">
                <div className="max-w-7xl mx-auto px-12 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-6">
                            <div className="text-4xl text-white">Artistic</div>
                            <div className="text-lg text-white hidden sm:block">ASD Detection System</div>
                        </div>
                        <div className="flex items-center gap-8">
                            <input type="hidden" id="apiUrl" defaultValue={apiUrl} />
                            <Link
                                href="/"
                                className="px-5 py-2 rounded-xl border border-lime-400 text-lime-300 hover:bg-lime-900 hover:text-white transition-all duration-200 text-sm font-medium"
                            >
                                Back
                            </Link>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-12 py-10">
                <div className="mb-8">
                    <h1 className="text-4xl font-semibold text-gray-900">Conversational Feature Guidelines</h1>
                    <p className="text-gray-600 mt-2 max-w-3xl">
                        This table documents all extracted conversational, linguistic, and interactional features used by the ASD detection models, along with their interpretations.
                    </p>
                </div>

                {/* Search */}
                <div className="mb-4">
                    <input
                        type="text"
                        id="searchInput"
                        placeholder="Search feature name or description..."
                        className="w-full md:w-1/2 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-lime-600"
                        onKeyUp={() => {
                            const input = document.getElementById('searchInput') as HTMLInputElement;
                            const filter = input.value.toLowerCase();
                            const rows = document.querySelectorAll('#featureTable tbody tr');
                            rows.forEach((row: Element) => {
                                const text = (row as HTMLElement).innerText.toLowerCase();
                                (row as HTMLElement).style.display = text.includes(filter) ? '' : 'none';
                            });
                        }}
                    />
                </div>

                {/* Table */}
                <div className="overflow-x-auto bg-white border rounded-2xl shadow-sm max-h-[500px] overflow-y-auto">
                    <table className="min-w-full text-sm text-left" id="featureTable">
                        <thead className="bg-lime-100 text-gray-800 sticky top-0 z-10">
                            <tr>
                                <th className="px-6 py-4 font-semibold">Feature Name</th>
                                <th className="px-6 py-4 font-semibold">Description</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y">
                            <tr>
                                <td className="px-6 py-4 font-mono text-gray-900">participant_id</td>
                                <td className="px-6 py-4 text-gray-700">Unique identifier assigned to each participant.</td>
                            </tr>
                            <tr>
                                <td className="px-6 py-4 font-mono text-gray-900">total_turns</td>
                                <td className="px-6 py-4 text-gray-700">Total number of conversational turns taken by the participant.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                {/* SHAP Explanations */}
                <section className="mt-16">
                    <div className="bg-white border rounded-2xl p-8 shadow-sm">
                        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Understanding Model SHAP Explanations</h2>
                        <p className="text-gray-600 mb-6 max-w-3xl">
                            These visualizations explain how conversational behaviors influence ASD or TD predictions.
                        </p>

                        {/* Toggle Buttons */}
                        <div className="flex gap-2 mb-8">
                            <button className="shap-tab active" onClick={(e) => {
                                document.querySelectorAll('.shap-panel').forEach(p => p.classList.add('hidden'));
                                document.querySelectorAll('.shap-tab').forEach(b => b.classList.remove('active'));
                                document.getElementById('shap-waterfall')?.classList.remove('hidden');
                                (e.target as HTMLElement).classList.add('active');
                            }}>Waterfall</button>
                            <button className="shap-tab" onClick={(e) => {
                                document.querySelectorAll('.shap-panel').forEach(p => p.classList.add('hidden'));
                                document.querySelectorAll('.shap-tab').forEach(b => b.classList.remove('active'));
                                document.getElementById('shap-bar')?.classList.remove('hidden');
                                (e.target as HTMLElement).classList.add('active');
                            }}>Bar Plot</button>
                            <button className="shap-tab" onClick={(e) => {
                                document.querySelectorAll('.shap-panel').forEach(p => p.classList.add('hidden'));
                                document.querySelectorAll('.shap-tab').forEach(b => b.classList.remove('active'));
                                document.getElementById('shap-beeswarm')?.classList.remove('hidden');
                                (e.target as HTMLElement).classList.add('active');
                            }}>Beeswarm</button>
                        </div>

                        {/* Waterfall */}
                        <div id="shap-waterfall" className="shap-panel">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src="/images/waterfall.png" className="rounded-xl border w-full max-w-4xl mx-auto mb-6" alt="SHAP Waterfall Plot" />
                            <div className="explain-box">
                                <h3>Waterfall Plot — Why THIS prediction was made</h3>
                                <p><strong>What it answers:</strong><br />
                                    &quot;Which behaviors pushed this prediction toward ASD or TD, and by how much?&quot;</p>
                                <ul>
                                    <li>Each row represents one conversational feature</li>
                                    <li><span className="red">Red bars</span> increase likelihood of ASD</li>
                                    <li><span className="blue">Blue bars</span> decrease likelihood (toward TD)</li>
                                    <li>Bars accumulate step-by-step to reach the final prediction score</li>
                                </ul>
                                <p><strong>What the numbers mean:</strong></p>
                                <ul>
                                    <li><strong>Numbers next to feature names</strong> (e.g. <code>−0.56</code>, <code>1.32</code>)<br />→ The child&apos;s <em>standardized value</em> for that feature compared to the training population</li>
                                    <li><strong>Numbers inside the bars</strong> (e.g. <code>+0.03</code>, <code>−0.02</code>)<br />→ How much that feature <em>shifted the model&apos;s decision</em> toward ASD or TD</li>
                                    <li><strong>Baseline value (E[f(X)])</strong><br />→ The model&apos;s average prediction before seeing this child&apos;s data</li>
                                    <li><strong>Final value (f(x))</strong><br />→ The model&apos;s final confidence score for this child after all features are considered</li>
                                </ul>
                                <p><strong>How to read it clinically:</strong></p>
                                <ul>
                                    <li>Start at the baseline (average child)</li>
                                    <li>Move feature by feature from top to bottom</li>
                                    <li>Observe which behaviors push the prediction most strongly</li>
                                    <li>Small bars indicate minor influence and are usually not clinically dominant</li>
                                </ul>
                                <div className="example">
                                    <strong>Example interpretation:</strong><br />
                                    Higher continuation markers and discourse markers increased ASD likelihood, while stronger semantic coherence reduced it.
                                </div>
                            </div>
                        </div>

                        {/* Bar Plot */}
                        <div id="shap-bar" className="shap-panel hidden">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src="/images/global_bar.png" className="rounded-xl border w-full max-w-4xl mx-auto mb-6" alt="SHAP Bar Plot" />
                            <div className="explain-box">
                                <h3>Bar Plot — Which features matter MOST overall</h3>
                                <p><strong>What it answers:</strong><br />
                                    &quot;Which conversational behaviors are most influential across the entire dataset?&quot;</p>
                                <ul>
                                    <li>Each bar represents one conversational feature</li>
                                    <li>Bar length shows <strong>average importance</strong> of that feature</li>
                                    <li>Longer bars = greater overall influence on predictions</li>
                                    <li>This plot does <strong>not</strong> show ASD vs TD direction</li>
                                </ul>
                                <p><strong>What the numbers mean:</strong></p>
                                <ul>
                                    <li><strong>Bar length (|SHAP value|)</strong><br />→ Average size of that feature&apos;s influence across all children</li>
                                    <li><strong>No negative or positive sign</strong><br />→ Direction is removed to focus on <em>importance</em>, not diagnosis</li>
                                    <li><strong>Features at the top</strong><br />→ Consistently important behaviors in the population</li>
                                </ul>
                                <p><strong>How to read it clinically:</strong></p>
                                <ul>
                                    <li>Use this plot to understand <strong>key behavioral markers</strong></li>
                                    <li>Helpful for screening, research, and feature validation</li>
                                    <li>Not intended for individual-level interpretation</li>
                                </ul>
                                <div className="example">
                                    <strong>Example interpretation:</strong><br />
                                    Semantic coherence and turn-taking consistency are strong global indicators across many transcripts.
                                </div>
                            </div>
                        </div>

                        {/* Beeswarm */}
                        <div id="shap-beeswarm" className="shap-panel hidden">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src="/images/global_beeswarm.png" className="rounded-xl border w-full max-w-4xl mx-auto mb-6" alt="SHAP Beeswarm Plot" />
                            <div className="explain-box">
                                <h3>Beeswarm Plot — How feature values affect predictions</h3>
                                <p><strong>What it answers:</strong><br />
                                    &quot;How do low vs high values of each behavior influence ASD likelihood?&quot;</p>
                                <ul>
                                    <li>Each dot represents one child or transcript</li>
                                    <li>Dots are spread horizontally by their influence</li>
                                    <li>Right side → pushes prediction toward ASD</li>
                                    <li>Left side → pushes prediction toward TD</li>
                                </ul>
                                <p><strong>What the numbers and colors mean:</strong></p>
                                <ul>
                                    <li><strong>X-axis position (SHAP value)</strong><br />→ Strength and direction of influence for that feature</li>
                                    <li><strong>Red dots</strong><br />→ Higher values of that feature</li>
                                    <li><strong>Blue dots</strong><br />→ Lower values of that feature</li>
                                    <li><strong>Vertical spread</strong><br />→ Variability across children</li>
                                </ul>
                                <p><strong>How to read it clinically:</strong></p>
                                <ul>
                                    <li>Look for patterns, not individual points</li>
                                    <li>Helps identify risk thresholds and variability</li>
                                    <li>Useful for understanding heterogeneous presentations</li>
                                    <li>More exploratory than diagnostic</li>
                                </ul>
                                <div className="example">
                                    <strong>Example interpretation:</strong><br />
                                    Higher filled-pause ratios consistently push predictions toward ASD, while lower values cluster near TD.
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
        </>
    );
}
