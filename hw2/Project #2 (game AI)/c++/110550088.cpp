#include "STcpClient_1.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include <map>


using namespace std;


/*
    input position (x,y) and direction
    output next node position on this direction
*/
vector<int> Next_Node(int pos_x, int pos_y, int direction) {
    vector<int> result;  //建立一個大小為 2 的 vector，用來儲存座標
    result.resize(2);
    if (pos_y % 2 == 1) {
        if (direction == 1) {
            result[0] = pos_x;
            result[1] = pos_y - 1;
        }
        else if (direction == 2) {
            result[0] = pos_x + 1;
            result[1] = pos_y - 1;
        }
        else if (direction == 3) {
            result[0] = pos_x - 1;
            result[1] = pos_y;
        }
        else if (direction == 4) {
            result[0] = pos_x + 1;
            result[1] = pos_y;
        }
        else if (direction == 5) {
            result[0] = pos_x;
            result[1] = pos_y + 1;
        }
        else if (direction == 6) {
            result[0] = pos_x + 1;
            result[1] = pos_y + 1;
        }
    }
    else {
        if (direction == 1) {
            result[0] = pos_x - 1;
            result[1] = pos_y - 1;
        }
        else if (direction == 2) {
            result[0] = pos_x;
            result[1] = pos_y - 1;
        }
        else if (direction == 3) {
            result[0] = pos_x - 1;
            result[1] = pos_y;
        }
        else if (direction == 4) {
            result[0] = pos_x + 1;
            result[1] = pos_y;
        }
        else if (direction == 5) {
            result[0] = pos_x - 1;
            result[1] = pos_y + 1;
        }
        else if (direction == 6) {
            result[0] = pos_x;
            result[1] = pos_y + 1;
        }
    }
    return result;
}


/*
    輪到此程式移動棋子
    mapStat : 棋盤狀態為 12*12矩陣, 0=可移動區域, -1=障礙, 1~2為玩家1~2佔領區域
    gameStat : 棋盤歷史順序
    return Step
    Step : 4 elements, [x, y, l, dir]
            x, y 表示要畫線起始座標
            l = 線條長度(1~3)
            dir = 方向(1~6),對應方向如下圖所示
              1  2
            3  x  4
              5  6
*/

bool check_legal_move(const int mapStat[12][12], int x, int y, int dir, int l){
    if (mapStat[x][y] == 0) {
        for(int i=0;i<l-1;i++){
            vector<int> next_node = Next_Node(x, y, dir);
            x = next_node[0];
            y = next_node[1];
            if(mapStat[x][y] != 0){
                return false;
            }
        }
        return true;
    }
    return false;
}

vector<vector<int>> get_legal_moves(const int mapStat[12][12]){
    // for(int i=0;i<12;i++){
    //     for(int j=0;j<12;j++){
    //         printf("%2d ", mapStat[i][j]);
    //     }
    //     cout << endl;
    // }
    vector<vector<int>> res;
    for(int i=0;i<12;i++){
        for(int j=0;j<12;j++){
            if(mapStat[i][j] == 0){
                vector<int> tmp = {i, j, 1, 1};
                res.push_back(tmp);
                
                for(int dir=1;dir<=6;dir++){
                    int x = i;
                    int y = j;
                    for(int l=2;l<=3;l++){
                        vector<int> next_node = Next_Node(x, y, dir);
                        x = next_node[0];
                        y = next_node[1];
                        if(mapStat[x][y] == 0){
                            vector<int> tmp1 = {i, j, l, dir};
                            res.push_back(tmp1);
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }
    // for(auto i: res){
    //     cout << i[0] << " " << i[1] << " " << i[2] << " " << i[3] << endl;
    // }
    return res;
}

class mctsNode {
public:
    mctsNode(const int _mapStat[12][12], const int player): parent(nullptr), chosen_action(), visit(0), win(0), expand_idx(0), end_state(false), player(player) {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                this->mapStat[i][j] = _mapStat[i][j];
            }
        }
    }
    mctsNode(mctsNode* parent, const vector<int> action, const int _mapStat[12][12], const bool end_state, const int player) : parent(parent), chosen_action(action), 
    visit(0), win(0), expand_idx(0), end_state(end_state), player(player) {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                this->mapStat[i][j] = _mapStat[i][j];
            }
        }
    }

    ~mctsNode() {
		for (auto& child : children)
			delete child;
	}

    double UCB1() {
        if (visit == 0) return 1e9;
		return (double)win / (double)visit + (double)1.41421 * sqrt((double)log(parent->visit) / double(visit));
    }

    mctsNode* parent;
    vector<int> chosen_action;
    int mapStat[12][12];
    int visit;
    int win;
    int expand_idx;
    bool end_state;
    int player;
    vector<mctsNode*> children;
};

const int T = 1000;
int cur_player = 1;

class MCTSPlayer {
public:
    MCTSPlayer(int player, int parallel, const int mapStat[12][12]) : player(player), parallel(parallel) {
        engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
        for (int i = 0; i < parallel; i++) {
            roots.push_back(new mctsNode(mapStat, (player == 1 ? 2 : 1)));
        }
    }

    ~MCTSPlayer() {
        for (auto& root : roots){
            delete root;
        }
    }

    vector<int> get_action(const int mapStat[12][12]) {
        vector<thread> threads;
        for (int i = 0; i < parallel; i++) {
            threads.push_back(thread(&MCTSPlayer::mcts, this, mapStat, i));
        }
        for (auto& thread : threads) {
            thread.join();
        }

        // mcts(mapStat, 0);
        vector<int> best_action = vector<int>();
		std::map<vector<int>, int> action_visit;

		for(mctsNode* root: roots){
			for(mctsNode* i: root->children){
				action_visit[i->chosen_action] += i -> visit;
			}
		}

		std::vector<std::pair<vector<int>, int>> action_visit_vec(action_visit.begin(), action_visit.end());
		shuffle(action_visit_vec.begin(), action_visit_vec.end(), engine);
		int max_visit = 0;
		for(auto const& i: action_visit_vec){
			if(i.second > max_visit){
				max_visit = i.second;
				best_action = i.first;
			}
		}

		return best_action;
    }

    void mcts(const int mapStat[12][12], int thread_idx){
        const auto threshold = std::chrono::milliseconds(T);
        int num_of_simulations = 13500;

        delete roots[thread_idx];
        roots[thread_idx] = new mctsNode(mapStat, (player == 1 ? 2 : 1));
        auto root = roots[thread_idx];

        auto start_time = std::chrono::high_resolution_clock::now();
        int cnt = 0;
        while(num_of_simulations -- && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) < threshold){
            cnt ++;
            mctsNode* node = root;
            while (node->children.size() != 0 && node->expand_idx == node->children.size()) {
                double max_ucb = -1e9;
                mctsNode* max_node = nullptr;
                for (auto& child : node->children) {
                    double ucb = child->UCB1();
                    if (ucb > max_ucb) {
                        max_ucb = ucb;
                        max_node = child;
                    }
                }
                node = max_node;
            }

            // expansion
            if (node->end_state == false && (int)node->children.size() == 0){
                auto legal_moves = get_legal_moves(node->mapStat);
                if(legal_moves.size() != 0){
                    shuffle(legal_moves.begin(), legal_moves.end(), engine);
                    for (auto& action: legal_moves){
                        int new_mapStat[12][12];
                        for (int i = 0; i < 12; i++) {
                            for (int j = 0; j < 12; j++) {
                                new_mapStat[i][j] = node->mapStat[i][j];
                            }
                        }
                        int x = action[0];
                        int y = action[1];
                        int l = action[2];
                        int dir = action[3];
                        new_mapStat[x][y] = (node->player == 1 ? 2 : 1);
                        for(int i=1;i<l;i++){
                            vector<int> next_node = Next_Node(x, y, dir);
                            x = next_node[0];
                            y = next_node[1];
                            new_mapStat[x][y] = (node->player == 1 ? 2 : 1);
                        }
                        node->children.push_back(new mctsNode(node, action, new_mapStat, false, (node->player == 1 ? 2 : 1)));
                    }
                } else {
                    node->end_state = true;
                }
            }

            // simulation
            int winner;
            if(node->end_state == false){
                int idx = node->expand_idx;
                node->expand_idx ++;
                node = node->children[idx];
                int player = (node->player == 1 ? 2 : 1);
                int cur_state[12][12];
                for (int i = 0; i < 12; i++) {
                    for (int j = 0; j < 12; j++) {
                        cur_state[i][j] = node->mapStat[i][j];
                    }
                }
                while(true){
                    auto legal_moves = get_legal_moves(cur_state);
                    if(legal_moves.size() == 0){
                        winner = player;
                        break;
                    }
                    shuffle(legal_moves.begin(), legal_moves.end(), engine);
                    int x = legal_moves[0][0];
                    int y = legal_moves[0][1];
                    int l = legal_moves[0][2];
                    int dir = legal_moves[0][3];
                    cur_state[x][y] = player;
                    for(int i=1;i<l;i++){
                        vector<int> next_node = Next_Node(x, y, dir);
                        x = next_node[0];
                        y = next_node[1];
                        cur_state[x][y] = player;
                    }
                    player = (player == 1 ? 2 : 1);
                }
            } else {
                winner = (node->player == 1 ? 2 : 1);
            }

            while(node != nullptr){
                node->visit ++;
                if (node->player == winner) node->win ++;
                node = node->parent;
            }
        }
    }


    int player = cur_player;    
    int parallel = 1;
    int T = 1500;
    vector<mctsNode*> roots;
    std::default_random_engine engine;
};



vector<int> GetStep(MCTSPlayer& player, int mapStat[12][12], int gameStat[12][12]) {
    vector<int> step = player.get_action(mapStat);
    return step;
}

int main()
{
    int id_package;
    int mapStat[12][12];
    int gameStat[12][12];
    MCTSPlayer player = MCTSPlayer(cur_player, 3, gameStat);

    while (true)
    {
        if (GetBoard(id_package, mapStat, gameStat))
            break;

        std::vector<int> step = GetStep(player, mapStat, gameStat);
        SendStep(id_package, step);
    }
}
