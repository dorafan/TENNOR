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

    enum { BUCKET_SIZE = 4 };
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







            for (int i = 0; i < BUCKET_SIZE; ++i) {
                int uid = blocks[i]->uid;
                bool is_dummy = obl::OblEq(uid, -1);

                tmp = obl::OblChoose(is_dummy, i, tmp);
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


        static inline void obl_copy_bytes(bool pred,
                                        const uint8_t *src,
                                        uint8_t       *dst,
                                        size_t         size) noexcept {

            pred = pred ? 1 : 0;

            size_t i = 0;
            while (i < size) {
                if (i + 8 <= size) {
                    uint64_t s_word = 0, d_word = 0, out = 0;
                    std::memcpy(&s_word, src + i, 8);
                    std::memcpy(&d_word, dst + i, 8);


                    obl::OblAssign<uint64_t>(pred, s_word, d_word, &out);

                    std::memcpy(dst + i, &out, 8);
                    i += 8;
                } else if (i + 4 <= size) {
                    uint32_t s_word = 0, d_word = 0, out = 0;
                    std::memcpy(&s_word, src + i, 4);
                    std::memcpy(&d_word, dst + i, 4);

                    obl::OblAssign<uint32_t>(pred, s_word, d_word, &out);

                    std::memcpy(dst + i, &out, 4);
                    i += 4;
                } else {
                    uint8_t s_byte = src[i];
                    uint8_t d_byte = dst[i];

                    obl::OblAssign<uint8_t>(pred, s_byte, d_byte, &dst[i]);
                    ++i;
                }
            }
        }




         static inline void obl_copy_block(bool pred,
                                  const Block *src,
                                  Block       *dst) noexcept {
            if (!src || !dst) {
                return;
            }


            pred = pred ? 1 : 0;


            dst->uid        = obl::OblChoose<int>(pred, src->uid,        dst->uid);
            dst->leaf       = obl::OblChoose<int>(pred, src->leaf,       dst->leaf);
            dst->encrypted  = obl::OblChoose<bool>(pred, src->encrypted, dst->encrypted);
            dst->cipher_len = obl::OblChoose<int>(pred, src->cipher_len, dst->cipher_len);


            const size_t IV_BYTES = IV_SIZE / BYTE_SIZE;
            obl_copy_bytes(pred,
                        reinterpret_cast<const uint8_t *>(src->iv),
                        reinterpret_cast<uint8_t *>(dst->iv),
                        IV_BYTES);


            obl_copy_bytes(pred,
                        reinterpret_cast<const uint8_t *>(src->ciphertext),
                        reinterpret_cast<uint8_t *>(dst->ciphertext),
                        BUFFER_SIZE);


            const std::string &s_data = src->data;
            std::string       &d_data = dst->data;

        #ifdef DEBUG

            assert(s_data.size() == d_data.size());
        #endif

            const size_t DATA_BYTES = d_data.size();


            const uint8_t *s_ptr = reinterpret_cast<const uint8_t *>(s_data.data());
            uint8_t       *d_ptr = reinterpret_cast<uint8_t *>(&d_data[0]);

            obl_copy_bytes(pred, s_ptr, d_ptr, DATA_BYTES);
        }










        std::map<int, int> position_map;
        std::vector<Block*> stash;

        unsigned char key[KEY_SIZE / BYTE_SIZE];


    	long long evictionTime;


    	Client() : evictionTime(0) {
	    position_map.clear();

    	}


        ~Client() {
            position_map.clear();

            for (int i = 0; i < stash.size(); i++) {
                delete stash[i];
            }
            stash.clear();
            stash.shrink_to_fit();
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
            int size = stash.size();

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









            for (auto &it : position_map) {
                bool is_match = obl::OblEq(it.first, uid);

                foundLeaf = obl::OblChoose(is_match, it.second, foundLeaf);


                (void)it.second;
            }
            return foundLeaf;
        }

        void setLeafOblivious(int uid, int newLeaf) {










            for (auto it = position_map.begin(); it != position_map.end(); ++it) {
                bool is_match = obl::OblEq(it->first, uid);
                int oldLeaf   = it->second;

                obl::OblAssign(is_match, newLeaf, oldLeaf, &it->second);
            }
        }



        std::string read(Server* s, int uid) {


            int oldLeaf = getLeafOblivious(uid);
            if (oldLeaf == -1) {

                return "";
            }


            fillStash(s, oldLeaf);
            Block* found = nullptr;














            Block scratch = *stash[0];

            for (Block *blk : stash) {
                bool is_match = obl::OblEq<int>(blk->uid, uid);

                obl_copy_block(is_match, blk, &scratch);
            }




            int newLeaf = randomLeaf();

            setLeafOblivious(uid, newLeaf);


            found->leaf = newLeaf;

            std::string store = found->data;
            evict(s, oldLeaf);
            return store;
        }






































































        void write(Server* s, int uid, const std::string& data) {

            int oldLeaf = getLeafOblivious(uid);






            int newLeaf = randomLeaf();
            setLeafOblivious(uid, newLeaf);


            fillStash(s, oldLeaf);


            for (Block* blk : stash) {
                bool is_match = obl::OblEq(blk->uid, uid);


                blk->leaf = obl::OblChoose(is_match, newLeaf, blk->leaf);


                size_t dst_len = blk->data.size();
                size_t src_len = data.size();
                size_t len = (dst_len < src_len) ? dst_len : src_len;



                obl_copy_bytes(is_match,
                               reinterpret_cast<const unsigned char*>(data.data()),
                               reinterpret_cast<unsigned char*>(&blk->data[0]),
                               len);
            }


            evict(s, oldLeaf);
        }












































        void evict(Server* s, int leaf) {
            auto start = std::chrono::high_resolution_clock::now();

            int path[PATHSIZE];
            getPath(leaf, path);


            const int stashSize = static_cast<int>(stash.size());

            Block dummy;


            for (int n = PATHSIZE - 1; n >= 0; --n) {
                int nodeID = path[n];
                Node* nodey = new Node();

                int filled = 0;


                for (int i = 0; i < stashSize; ++i) {
                    Block* curr = stash[i];


                    int common = commonAncestor(curr->leaf, leaf);

                    bool on_path  = obl::OblLe(nodeID, common);
                    bool is_real  = !obl::OblEq(curr->uid, -1);
                    bool not_full = obl::OblLt(filled, static_cast<int>(BUCKET_SIZE));


                    bool can_use = on_path & is_real & not_full;


                    for (int slot = 0; slot < BUCKET_SIZE; ++slot) {
                        bool is_target_slot = obl::OblEq(slot, filled);
                        bool place_here     = can_use & is_target_slot;



                        obl_copy_block(place_here,
                                       curr,
                                       nodey->bucket->blocks[slot]);
                    }


                    obl_copy_block(can_use, &dummy, curr);


                    int filled_plus_one = filled + 1;
                    filled = obl::OblChoose(can_use, filled_plus_one, filled);
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
