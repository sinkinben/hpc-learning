#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <algorithm>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//

void pageRank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
    Vertex *sink_nodes = new Vertex[numNodes];
    int num_sinks = 0;

    /* collect all vertices with no outgoing edges, and
     * init solution[i] with equal probability
     */
    for (int i = 0; i < numNodes; ++i)
    {
        solution[i] = equal_prob;
        if (outgoing_size(g, i) == 0)
            sink_nodes[num_sinks++] = i;
    }


    bool converged = false;
    double *score_new = new double [numNodes];
    double term = (1.0 - damping) / numNodes;

    while (!converged)
    {
        /* traverse vertices with no outgoing edges */
        double sink_val = 0;
        for (int i = 0; i < num_sinks; ++i)
        {
            int u = sink_nodes[i];
            sink_val += solution[u];
        }
        sink_val = sink_val * damping / numNodes;

        for (int v = 0; v < numNodes; ++v)
        {
            /* result of score_new[v] */
            double val = 0.0;

            /* traverse all edges of (u, v) */
            const Vertex *end = incoming_end(g, v);
            for (const Vertex *u = incoming_begin(g, v); u != end; ++u)
                val += solution[*u] / outgoing_size(g, *u);
            val = term + damping * val + sink_val;

            score_new[v] = val;
        }

        double diff = 0.0;
        for (int i = 0; i < numNodes; ++i)
        {
            diff += std::abs(score_new[i] - solution[i]);
            solution[i] = score_new[i];
        }

        converged = (diff < convergence);
    }
    delete[] sink_nodes;
    delete[] score_new;
}
