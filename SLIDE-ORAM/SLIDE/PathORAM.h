#ifndef PATHORAM_H
#define PATHORAM_H

#include <stdint.h>
#include <iostream>
#include <iterator>
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <cstring>
#include <bitset>
#include <cstdlib>

#include <chrono>

#include "Config.h"

#include "obl.h"




template <int N_value>
class PathORAM {
public:
    PathORAM() {
        client = new Client();
        server = new Server();
        client->initServer(server);
    }

    ~PathORAM() {
        delete client;
        delete server;
    }

    void init() {
        client->initServer(server);
    }

    std::string read(int uid) {
        return client->read(server, uid);
    }

    void write(int uid, const std::string& data) {
        client->write(server, uid, data);
    }


    long long getEvictionTime() const {
        return client->evictionTime;
    }


    static const int computeLog2(int n) {
        return (n <= 1) ? 0 : 1 + computeLog2(n / 2);
    }


    static const int N = N_value;
    static const int numNodes = N * 2 - 1;
    static const int BUCKET_SIZE = 4;
    static const std::string DUMMY;


    static const int PATHSIZE;

    static const int KEY_SIZE = 256;
    static const int IV_SIZE = 128;
    static const int BYTE_SIZE = 8;
    static const int BUFFER_SIZE = 2 * 1024;
    static const bool ENCRYPTION = false;

private:


    class Block {
    public:
        int uid;
        int leaf;
        std::string data;

        bool encrypted;
        unsigned char iv[IV_SIZE / BYTE_SIZE];
        unsigned char ciphertext[BUFFER_SIZE];
        int cipher_len;

        Block() {
            uid = -1;
            leaf = -1;
            data = DUMMY;

            emptyArray(iv, IV_SIZE / BYTE_SIZE);
            encrypted = false;
            emptyArray(ciphertext, BUFFER_SIZE);
            cipher_len = 0;
        }

        Block(int uidP, int leafP, const std::string& dataP) {
            uid = uidP;
            leaf = leafP;
            data = dataP;

            emptyArray(iv, IV_SIZE / BYTE_SIZE);
            encrypted = false;
            emptyArray(ciphertext, BUFFER_SIZE);
            cipher_len = 0;
        }

        void makePayload() {
            std::string uid_string = std::to_string(uid);
            std::string leaf_string = std::to_string(leaf);
            std::string payload = uid_string + "," + leaf_string + "," + data;

            data = payload;
        }

        void breakPayload() {
            std::string delimiter = ",";

            size_t pos = 0;
            pos = data.find(delimiter);
            std::string uid_string = data.substr(0, pos);
            data.erase(0, pos + delimiter.length());
            uid = std::stoi(uid_string);

            pos = 0;
            pos = data.find(delimiter);
            std::string leaf_string = data.substr(0, pos);
            data.erase(0, pos + delimiter.length());
            leaf = std::stoi(leaf_string);

            emptyArray(iv, IV_SIZE / BYTE_SIZE);
            encrypted = false;
            emptyArray(ciphertext, BUFFER_SIZE);
            cipher_len = 0;
        }

        void string2UnsignedChar(unsigned char store[]) {
            std::strcpy((char*)store, data.c_str());
        }

        void easy_encrypt(unsigned char* key) {
            if (ENCRYPTION) {

            }
        }

        void easy_decrypt(unsigned char* key) {
            if (ENCRYPTION) {

            }
        }

        void printBlock() {
            if (encrypted) {

            }
            else {
                std::cout << "BLOCK ID: " << uid << " LEAF: " << leaf << " DATA: " << data << std::endl;
            }
        }

    private:
        void emptyArray(unsigned char emptying[], int size) {
            std::fill_n(emptying, size, 0x00);
        }
    };

    class Bucket {
    public:
        Block* blocks[BUCKET_SIZE];

        Bucket() {
            for (int i = 0; i < BUCKET_SIZE; i++) {
                blocks[i] = new Block();
            }
        }

        ~Bucket() {
            for (int i = 0; i < BUCKET_SIZE; i++) {
                delete blocks[i];
            }
        }

        int findDummyIndex() {

            int tmp = -1;
            for (int i = 0; i < BUCKET_SIZE; i++) {
                if (blocks[i]->uid == -1) {

                    tmp = i;
                }
            }

            return tmp;
        }

        void encryptBucket(unsigned char* key) {
            for (int i = 0; i < BUCKET_SIZE; i++) {
                blocks[i]->easy_encrypt(key);
            }
        }

        void decryptBucket(unsigned char* key) {
            for (int i = 0; i < BUCKET_SIZE; i++) {
                blocks[i]->easy_decrypt(key);
            }
        }

        void printBucket() {
            std::cout << "\n--- BUCKET w/ size " << BUCKET_SIZE << " --- " << std::endl;
            for (int i = 0; i < BUCKET_SIZE; i++) {
                blocks[i]->printBlock();
            }
            std::cout << "--------------------------" << std::endl;
        }
    };

    class Node {
    public:
        Bucket* bucket;
        bool isLeaf;

        Node(int index) {
            isLeaf = checkLeaf(index);
            bucket = new Bucket();
        }

        Node() {
            isLeaf = false;
            bucket = new Bucket();
        }

        ~Node() {
            delete bucket;
        }

        void printNode(int index) {
            std::cout << "\n --- NODE " << index << " --- " << std::endl;
            std::cout << "Left Index: " << left(index) << " Right Index: " << right(index) << " Parent Index: " << parent(index) << std::endl;
            bucket->printBucket();
        }
    };

    class Tree {
    public:
        int levels;
        Node* nodes[numNodes];

        Tree() {
            levels = (int)std::log2(N);
            for (int i = 0; i < numNodes; i++) {
                nodes[i] = new Node(i);
            }
        }

        ~Tree() {
            for (int i = 0; i < numNodes; i++) {
                delete nodes[i];
            }
        }

        void printTree() {
            std::cout << "\n --- TREE of size " << numNodes << " --- " << std::endl;

            for (int i = 0; i < numNodes; i++) {
                nodes[i]->printNode(i);
            }
        }
    };

    class Server {
    public:
        Tree* tree;

        Server() {
            tree = new Tree();
        }

        ~Server() {
            delete tree;
        }

        void printServer() {
            std::cout << " ----- SERVER STATE START ----- " << std::endl;
            tree->printTree();
            std::cout << " ------ SERVER STATE END ------ " << std::endl;
        }
    };

    class Client {
    public:
        std::map<int, int> position_map;
        std::vector<Block*> stash;

        unsigned char key[KEY_SIZE / BYTE_SIZE];


        long long evictionTime;


        Client() : evictionTime(0) {
            position_map.clear();

        }


        ~Client() {
            position_map.clear();

            for (int i = 0; i < (int)stash.size(); i++) {
                delete stash[i];
            }
            stash.clear();
            stash.shrink_to_fit();
        }


        static void simulatedObliviousCopySize(size_t units) {












            if (units == 0) {
                units = 1;
            }


            volatile int acc = 0;


            int x = 0x12345678;
            int y = 0x9abcdef0;

            for (size_t i = 0; i < units; ++i) {

                int cond = static_cast<int>(i & 1);


                int chosen = obl::OblChoose(cond, x, y);


                acc ^= chosen;
                x ^= chosen;
                y += cond;
            }

            (void)acc;
        }

        static void simulatedObliviousCopyString(const std::string &s) {
            simulatedObliviousCopySize(s.size());
        }

        static void simulatedObliviousCopyBlock(const Block *b) {
            if (b == nullptr) {
                simulatedObliviousCopySize(1);
                return;
            }

            size_t totalBytes = b->data.size()
                              + (IV_SIZE / BYTE_SIZE)
                              + BUFFER_SIZE;
            simulatedObliviousCopySize(totalBytes);
        }


        void initServer(Server* s) {
            for (int i = 0; i < numNodes; i++) {
                Node* n = new Node();
                n->bucket->encryptBucket(key);
                delete s->tree->nodes[i];
                s->tree->nodes[i] = n;
            }
        }

        void printClient() {
            std::cout << "\n --- Client Position Map ---" << std::endl;
            std::map<int, int>::iterator itr;
            std::cout << "\tUID\tLEAF\n";
            for (itr = position_map.begin(); itr != position_map.end(); ++itr) {
                std::cout << '\t' << itr->first << '\t' << itr->second
                    << '\n';
            }

            std::cout << "\n --- Client Stash ---" << std::endl;
            for (Block* i : stash) {
                i->printBlock();
            }

            std::cout << " --------------------------" << std::endl;
        }

        Node* fetch(Server* s, int nid) {
            Node* dummy = new Node();
            dummy->bucket->encryptBucket(key);

            Node* wanted = s->tree->nodes[nid];
            wanted->bucket->decryptBucket(key);

            s->tree->nodes[nid] = dummy;
            return wanted;
        }

        void deleteStashDummys() {
            int size = (int)stash.size();

            Block* check;

            for (int i = 0; i < size; i++) {
                check = stashGetBack();
                if (check->uid == -1) {
                    delete check;
                }
                else {
                    stashPutFront(check);
                }
            }

            stash.shrink_to_fit();
        }

        void fillStash(Server* s, int leaf) {
            int path[PATHSIZE];
            getPath(leaf, path);

            for (int i = 0; i < PATHSIZE; i++) {
                Node* current = fetch(s, path[i]);
                for (int z = 0; z < BUCKET_SIZE; z++) {
                    Block* original = current->bucket->blocks[z];
                    Block* copy = new Block(original->uid, original->leaf, original->data);
                    stash.push_back(copy);
                }
                delete current;
            }

            deleteStashDummys();
        }

        Block* stashGetBack() {
            Block* check = stash.back();
            stash.pop_back();

            return check;
        }

        void stashPutFront(Block* newFront) {
            stash.insert(stash.begin(), newFront);
        }





        int getLeafOblivious(int uid) {
            int foundLeaf = -1;

            for (auto & it : position_map) {
                if (it.first == uid) {
                    foundLeaf = it.second;
                } else {

                    (void) it.second;
                }

                int key = it.first;
                int value = it.second;

                int pred = obl::OblEq(key, uid);
                int tmp = obl::OblChoose(pred, value, value);
            }


            return foundLeaf;
        }

        void setLeafOblivious(int uid, int newLeaf) {

            for (auto it = position_map.begin(); it != position_map.end(); ++it) {
                if (it->first == uid) {
                    it->second = newLeaf;
                } else {

                    (void) it->second;
                }

                int key = it->first;
                int value = it->second;

                int pred = obl::OblEq(key, uid);
                int tmp = obl::OblChoose(pred, value, value);
            }
        }



        std::string read(Server* s, int uid) {


            int oldLeaf = getLeafOblivious(uid);
            if (oldLeaf == -1) {

                return "";
            }


            fillStash(s, oldLeaf);
            Block* found = nullptr;

            for (Block* i : stash) {
                if (i->uid == uid) {
                    found = i;

                }
            }

            if (found == nullptr) {

                return "";
            }

            int newLeaf = randomLeaf();

            setLeafOblivious(uid, newLeaf);


            found->leaf = newLeaf;


            for (Block* i : stash) {
                int tmp = obl::OblEq(i->uid, uid);
                simulatedObliviousCopyBlock(found);
            }

            std::string store = found->data;
            evict(s, oldLeaf);
            return store;
        }

        void write(Server* s, int uid, const std::string& data) {

            for (Block* i : stash){
                 int tmp = obl::OblEq(i->uid, uid);
                simulatedObliviousCopyString(data);
            }

            bool alreadyExists = false;
            {


                for (auto it = position_map.begin(); it != position_map.end(); ++it) {
                    if (it->first == uid) {
                        alreadyExists = true;
                    }
                    (void) it->second;
                }
            }

            if (!alreadyExists) {
                position_map[uid] = randomLeaf();
                int leaf = position_map[uid];

                fillStash(s, leaf);

                Block* alpha = new Block(uid, leaf, data);
                stash.push_back(alpha);
            }
            else{
                int oldLeaf = getLeafOblivious(uid);
                int newLeaf = randomLeaf();
                setLeafOblivious(uid, newLeaf);
                fillStash(s, oldLeaf);

                for (Block* i : stash) {
                    if (i->uid == uid) {
                        i->data = data;
                        i->leaf = newLeaf;

                    }
                }
            }

            evict(s, position_map[uid]);
        }

        void evict(Server* s, int leaf) {
            auto start = std::chrono::high_resolution_clock::now();

            int path[PATHSIZE];
            getPath(leaf, path);

            for (int n = PATHSIZE - 1; n >= 0; n--) {
                int nodeID = path[n];
                Node* nodey = new Node();
                int fillable = 0;

                int size = (int)stash.size();
                for (int i = 0; i < size; i++) {
                    Block* curr = stashGetBack();


                    simulatedObliviousCopyBlock(curr);

                    int common = commonAncestor(curr->leaf, leaf);

                    if (nodeID <= common) {
                        delete nodey->bucket->blocks[fillable];
                        nodey->bucket->blocks[fillable] = curr;
                        fillable++;
                    }
                    else {
                        stashPutFront(curr);
                    }

                    if (fillable == BUCKET_SIZE) {
                        break;
                    }
                }

                delete s->tree->nodes[nodeID];
                nodey->bucket->encryptBucket(key);
                s->tree->nodes[nodeID] = nodey;
            }

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            evictionTime += duration.count();
        }



    };


    static int smallLeaf() {
        return N - 1;
    }

    static int bigLeaf() {
        return 2 * N - 2;
    }

    static int randomLeaf() {
        return smallLeaf() + rand() % (bigLeaf() - smallLeaf() + 1);
    }

    static int randomRange(int small, int big) {
        return small + rand() % (big - small + 1);
    }

    static void levelRange(int level, int range[]) {
        int low = (int)pow(2, level) - 1;
        int high = (int)pow(2, level + 1) - 2;
        range[0] = low;
        range[1] = high;
    }

    static void pick2(int small, int big, int random[]) {
        random[0] = randomRange(small, big);
        random[1] = randomRange(small, big);
        while (random[0] == random[1]) {
            random[1] = randomRange(small, big);
        }
    }

    static bool checkLeaf(int index) {
        return (index >= smallLeaf()) && (index <= bigLeaf());
    }

    static int left(int index) {
        if (checkLeaf(index)) {
            return -1;
        }
        return (2 * index) + 1;
    }

    static int right(int index) {
        if (checkLeaf(index)) {
            return -1;
        }
        return (2 * index) + 2;
    }

    static int parent(int index) {
        if (index == 0) {
            return -1;
        }
        return (index - 1) / 2;
    }

    static void getPath(int leafid, int path[]) {
        int backwards = PATHSIZE - 1;
        int currentNode = leafid;

        while (backwards >= 0) {
            path[backwards] = currentNode;
            backwards--;
            currentNode = parent(currentNode);
        }
    }

    static int commonAncestor(int leaf1, int leaf2) {
        if (leaf1 == leaf2) {
            return leaf1;
        }
        else {
            return commonAncestor(parent(leaf1), parent(leaf2));
        }
    }


    Server* server;
    Client* client;
};


template <int N_value>
const std::string PathORAM<N_value>::DUMMY = "Dummy";

template <int N_value>
const int PathORAM<N_value>::PATHSIZE = (int)(std::log2(PathORAM::N) + 1);


#endif
