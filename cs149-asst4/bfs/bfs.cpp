#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <algorithm>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
#pragma omp parallel
    {
        /* thread local frontier */
        int *local_frontier = new int[g->num_nodes];
        int local_cnt = 0;

#pragma omp for
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                /* if dis[outgoing] != NOT_VISITED_MARKER, then we can reduce the number of CAS. */
                if (distances[outgoing] == NOT_VISITED_MARKER &&
                    __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                {
                    local_frontier[local_cnt++] = outgoing;
                }
            }
        }
        int offset = __sync_fetch_and_add(&new_frontier->count, local_cnt);
        memcpy(new_frontier->vertices + offset, local_frontier, local_cnt * sizeof(int));
        delete[] local_frontier;
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; ++i)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g, vertex_set *frontier, vertex_set *new_frontier,
    int *distances, int num_itorations)
{
#pragma omp parallel
    {
        /* thread local */
        int *local_frontier = new int[g->num_nodes];
        int local_cnt = 0;

#pragma omp for schedule(dynamic, 200)
        for (int v = 0; v < g->num_nodes; ++v)
        {
            /* traverse all vertices that are not visited */
            if (distances[v] != NOT_VISITED_MARKER)
                continue;

            /* traverse all incoming edges of v */
            const Vertex *end = incoming_end(g, v);
            for (auto ptr = incoming_begin(g, v); ptr != end; ++ptr)
            {
                int u = *ptr;
                /* if u is in current frontier set, then add v into new_frontier */
                if (distances[u] == num_itorations)
                {
                    distances[v] = num_itorations + 1;
                    local_frontier[local_cnt++] = v;
                    break;
                }
            }
        }

        int offset = __sync_fetch_and_add(&new_frontier->count, local_cnt);
        memcpy(new_frontier->vertices + offset, local_frontier, sizeof(int) * local_cnt);
        delete[] local_frontier;
    }
}

/* For more details of Bottom Up BFS, refer to:
 * https://people.csail.mit.edu/jshun/6886-s18/lectures/lecture4-1.pdf
 */
void bfs_bottom_up(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

// initialize all nodes to NOT_VISITED
#pragma omp for
    for (int i = 0; i < graph->num_nodes; ++i)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int num_iterations = 0;
    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances, num_iterations);
        num_iterations += 1;

        // swap pointers
        std::swap(new_frontier, frontier);
    }
}

void bfs_hybrid(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp for
    for (int i = 0; i < graph->num_nodes; ++i)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int num_iterations = 0;
    constexpr int threshold = int(1e7);
    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);
        if (frontier->count < threshold)
            top_down_step(graph, frontier, new_frontier, sol->distances);
        else
            bottom_up_step(graph, frontier, new_frontier, sol->distances, num_iterations);
        num_iterations += 1;

        // swap pointers
        std::swap(new_frontier, frontier);
    }
}
