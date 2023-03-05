#include "include/go.h"

namespace py = pybind11;

template <typename ... Ts, std::size_t ... Is>
std::tuple<Ts...> sumT (std::tuple<Ts...> const & t1,
                        std::tuple<Ts...> const & t2,
                        std::index_sequence<Is...> const &)
{
    return { (std::get<Is>(t1) + std::get<Is>(t2))... };
}

 template <typename ... Ts, std::size_t ... Is>
std::tuple<Ts...> minusT (std::tuple<Ts...> const & t1,
                        std::tuple<Ts...> const & t2,
                        std::index_sequence<Is...> const &)
{
    return { (std::get<Is>(t1) + std::get<Is>(t2))... };
}

template <typename ... Ts>
std::tuple<Ts...> operator+ (std::tuple<Ts...> const & t1,
                             std::tuple<Ts...> const & t2)
{
    return sumT(t1, t2, std::make_index_sequence<sizeof...(Ts)>{});
}

template <typename ... Ts>
std::tuple<Ts...> operator- (std::tuple<Ts...> const & t1,
                             std::tuple<Ts...> const & t2)
{
    return minusT(t1, t2, std::make_index_sequence<sizeof...(Ts)>{});
}


std::vector<int> go(const std::vector<std::pair<int, int>>&edges, int nnodes)
{
    boost::adjacency_list<> graph{edges.begin(), edges.end(), nnodes};

    std::vector<std:tuple<std::tuple<int, int>, int>> node_priorities;
    for (boost::tie(i, end) = vertices(g); i != end; ++i) {
        auto id = vertex_idMap[*i]
        node_priorities.push_back(
            std::make_tuple(
                std::make_tuple(-1 * boost::out_degree(id, graph), 0),
                id
            )
        );
    }

    PriorityQueue<std::tuple<int, int>, int> queue(node_priorities, std::make_tuple(0,0));
    std::vector<int> order;
    for(int i = 0; i < nnodes; i++){
        int node_id = queue.pop();
        order.push_back(node_id);
        boost::adjacency_list<>::out_edge_iterator eit, eend;
        std::tie(eit, eend) = boost::out_edges(topLeft, g);
        std::for_each(eit, eend,
            [&g](boost::adjacency_list<>::edge_descriptor it)
            { std::cout << boost::target(it, g) << '\n'; });
    }

    return 0;
}

PYBIND11_MODULE(go, m)
{
    m.doc() = "A module for the go algorithm";
    m.def("testPriorityQueue", &go);

}