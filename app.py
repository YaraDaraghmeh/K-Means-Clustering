from flask import Flask, render_template, request, jsonify
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)


class KMeansDemo:
    def __init__(self):
        self.data_points = []
        self.centroids = []
        self.labels = []
        self.history = []
        self.iteration = 0
        self.converged = False

    def generate_random_data(self, n_points=30, n_centers=3):
        """Generate random clustered data"""
        X, _ = make_blobs(n_samples=n_points, centers=n_centers,
                          cluster_std=1.5, random_state=42)
        # Scale to fit canvas coordinates (0-400)
        X = (X - X.min()) / (X.max() - X.min()) * 350 + 25
        self.data_points = X.tolist()
        self.reset_clustering()
        return self.data_points

    def initialize_centroids(self, k, method='random'):
        """Initialize centroids using random or k-means++ method"""
        if not self.data_points:
            return []

        data = np.array(self.data_points)

        if method == 'random':
            # Random initialization
            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            self.centroids = np.random.uniform(min_vals, max_vals, (k, 2)).tolist()
        else:
            # K-means++ initialization
            centroids = []
            # Choose first centroid randomly
            centroids.append(data[np.random.randint(len(data))].tolist())

            for _ in range(1, k):
                distances = []
                for point in data:
                    min_dist = min([np.linalg.norm(point - np.array(c)) for c in centroids])
                    distances.append(min_dist)

                # Choose next centroid with probability proportional to squared distance
                distances = np.array(distances) ** 2
                probabilities = distances / distances.sum()
                cumulative = probabilities.cumsum()
                r = np.random.rand()

                for i, cum_prob in enumerate(cumulative):
                    if r <= cum_prob:
                        centroids.append(data[i].tolist())
                        break

            self.centroids = centroids

        return self.centroids

    def assign_points_to_clusters(self):
        """Assign each point to the nearest centroid"""
        if not self.data_points or not self.centroids:
            return []

        data = np.array(self.data_points)
        centroids = np.array(self.centroids)

        labels = []
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            labels.append(np.argmin(distances))

        self.labels = labels
        return labels

    def update_centroids(self):
        """Update centroids to the mean of assigned points"""
        if not self.data_points or not self.labels:
            return False

        data = np.array(self.data_points)
        new_centroids = []
        centroids_changed = False

        for i in range(len(self.centroids)):
            cluster_points = data[np.array(self.labels) == i]
            if len(cluster_points) > 0:
                new_centroid = cluster_points.mean(axis=0).tolist()
            else:
                new_centroid = self.centroids[i]  # Keep old centroid if no points assigned

            # Check if centroid moved significantly
            if np.linalg.norm(np.array(new_centroid) - np.array(self.centroids[i])) > 0.1:
                centroids_changed = True

            new_centroids.append(new_centroid)

        self.centroids = new_centroids
        return centroids_changed

    def run_kmeans_step(self):
        """Run one iteration of k-means"""
        self.assign_points_to_clusters()
        centroids_changed = self.update_centroids()
        self.iteration += 1

        # Store history
        self.history.append({
            'iteration': self.iteration,
            'centroids': [c.copy() for c in self.centroids],
            'labels': self.labels.copy(),
            'inertia': self.calculate_inertia()
        })

        if not centroids_changed:
            self.converged = True

        return not centroids_changed

    def run_full_kmeans(self, k, max_iterations=20, init_method='random'):
        """Run complete k-means algorithm"""
        self.reset_clustering()
        self.initialize_centroids(k, init_method)

        for _ in range(max_iterations):
            if self.run_kmeans_step():
                break

        return self.get_results()

    def calculate_inertia(self):
        """Calculate within-cluster sum of squares (WCSS)"""
        if not self.data_points or not self.centroids or not self.labels:
            return 0

        data = np.array(self.data_points)
        centroids = np.array(self.centroids)

        inertia = 0
        for i, point in enumerate(data):
            if i < len(self.labels):
                cluster_id = self.labels[i]
                if cluster_id < len(centroids):
                    inertia += np.linalg.norm(point - centroids[cluster_id]) ** 2

        return round(inertia, 2)

    def reset_clustering(self):
        """Reset clustering state"""
        self.centroids = []
        self.labels = []
        self.history = []
        self.iteration = 0
        self.converged = False

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, list):
            return [KMeansDemo.to_serializable(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: KMeansDemo.to_serializable(v) for k, v in obj.items()}
        else:
            return obj

    def get_results(self):
        return KMeansDemo.to_serializable({
            'data_points': self.data_points,
            'centroids': self.centroids,
            'labels': self.labels,
            'iteration': self.iteration,
            'converged': self.converged,
            'inertia': self.calculate_inertia(),
            'history': self.history
        })

    def generate_plot(self):
        """Generate matplotlib plot as base64 string"""
        if not self.data_points:
            return None

        plt.figure(figsize=(8, 6))
        data = np.array(self.data_points)

        # Color palette
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        # Plot data points
        if self.labels:
            for i in range(len(self.centroids)):
                cluster_points = data[np.array(self.labels) == i]
                if len(cluster_points) > 0:
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                                c=colors[i % len(colors)], alpha=0.6, s=50,
                                label=f'Cluster {i}')
        else:
            plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.6, s=50, label='Unassigned')

        # Plot centroids
        if self.centroids:
            centroids = np.array(self.centroids)
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        c='black', marker='x', s=200, linewidths=3, label='Centroids')

        plt.title(f'K-Means Clustering - Iteration {self.iteration}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return plot_url


# Global instance
kmeans_demo = KMeansDemo()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_data', methods=['POST'])
def generate_data():
    data = request.get_json()
    n_points = data.get('n_points', 30)
    n_centers = data.get('n_centers', 3)

    points = kmeans_demo.generate_random_data(n_points, n_centers)
    return jsonify({
        'success': True,
        'data_points': points
    })


@app.route('/initialize_centroids', methods=['POST'])
def initialize_centroids():
    data = request.get_json()
    k = data.get('k', 3)
    method = data.get('method', 'random')

    centroids = kmeans_demo.initialize_centroids(k, method)
    return jsonify({
        'success': True,
        'centroids': centroids
    })


@app.route('/run_step', methods=['POST'])
def run_step():
    converged = kmeans_demo.run_kmeans_step()
    results = kmeans_demo.get_results()

    return jsonify({
        'success': True,
        'converged': converged,
        'results': results
    })


@app.route('/run_full', methods=['POST'])
def run_full():
    data = request.get_json()
    k = data.get('k', 3)
    max_iterations = data.get('max_iterations', 20)
    init_method = data.get('init_method', 'random')

    results = kmeans_demo.run_full_kmeans(k, max_iterations, init_method)

    return jsonify({
        'success': True,
        'results': results
    })


@app.route('/reset', methods=['POST'])
def reset():
    kmeans_demo.reset_clustering()
    return jsonify({'success': True})


@app.route('/get_plot')
def get_plot():
    plot_url = kmeans_demo.generate_plot()
    return jsonify({
        'success': True,
        'plot': plot_url
    })


@app.route('/add_point', methods=['POST'])
def add_point():
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')

    if x is not None and y is not None:
        kmeans_demo.data_points.append([float(x), float(y)])
        kmeans_demo.reset_clustering()

        return jsonify({
            'success': True,
            'data_points': kmeans_demo.data_points
        })

    return jsonify({'success': False, 'error': 'Invalid coordinates'})


@app.route('/compare_methods', methods=['POST'])
def compare_methods():
    """Compare different initialization methods"""
    data = request.get_json()
    k = data.get('k', 3)
    max_iterations = data.get('max_iterations', 20)

    results = {}

    # Test random initialization
    kmeans_demo.reset_clustering()
    random_results = kmeans_demo.run_full_kmeans(k, max_iterations, 'random')
    results['random'] = random_results

    # Test k-means++ initialization
    kmeans_demo.reset_clustering()
    kmeanspp_results = kmeans_demo.run_full_kmeans(k, max_iterations, 'kmeans++')
    results['kmeans++'] = kmeanspp_results

    return jsonify({
        'success': True,
        'comparison': results
    })


if __name__ == '__main__':
    app.run(debug=True)