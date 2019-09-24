#!usr/bin/env python
# coding=utf-8


class Node(object):
    def __init__(self, data=None, lchild=None, rchild=None, parent=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild
        self.parent = parent
# 创建一个节点类，包括数据域和两个分支


class BTree(object):
    global max_depth
    max_depth = 0

    def __init__(self):
        self.root = Node()
        self.queue = []      # 用于存放没有child的节点的队列，利用顺序进行树的插入

    def add_node(self, data):
        # 添加节点
        node = Node(data)
        if self.root.data is None:
            self.root = node
            self.queue.append(self.root)    # 将没有子节点的root存入队列
        else:
            treeNode = self.queue[0]   # 读取队列中的第一个节点
            if treeNode.lchild is None:
                treeNode.lchild = node
                treeNode.lchild.parent = treeNode
                self.queue.append(treeNode.lchild)
            else:
                treeNode.rchild = node
                treeNode.rchild.parent = treeNode
                self.queue.append(treeNode.rchild)
                self.queue.pop(0)    # 移除第一个节点元素，此时该节点已经有左分支和右分支
    # 添加节点的顺序类似于层次遍历

    def dlr(self, root):
        if root is None:
            return
        print(root.data, end=' ')
        self.dlr(root.lchild)
        self.dlr(root.rchild)
    # 先序遍历，data->lchild->rchlid

    def ldr(self, root):
        if root is None:
            return
        self.ldr(root.lchild)
        print(root.data, end=' ')
        self.ldr(root.rchild)
    # 中序遍历，lchild->data->rchlid

    def lrd(self, root):
        if root is None:
            return
        self.lrd(root.lchild)
        self.lrd(root.rchild)
        print(root.data, end=' ')
    # 后序遍历，lchild->rchlid->data

    def dlr_stack(self, root):
        if root is None:
            return
        stack = []
        node = root
        while node or stack:
            while node:
                print(node.data, end=' ')
                stack.append(node)
                node = node.lchild
            node = stack.pop()      # 移除最后一个元素并返回元素的值
            node = node.rchild

    def lrd_stack(self, root):
        if root is None:
            return
        stack1 = []
        stack2 = []
        node = root
        stack1.append(node)
        while stack1:
            node = stack1.pop()
            if node.lchild:
                stack1.append(node.lchild)
            if node.rchild:
                stack1.append(node.rchild)
            stack2.append(node)       # node的取出顺序为d->r->l,再存入堆栈就是lrd后序遍历
        while stack2:
            print(stack2.pop().data, end=' ')
        pass

    def lv(self, root):
        # 层次遍历
        if root is None:
            return
        queue = []
        node = root
        queue.append(node)
        while queue:
            node = queue.pop(0)
            print(node.data, end=' ')
            if node.lchild is not None:
                queue.append(node.lchild)
            if node.rchild is not None:
                queue.append(node.rchild)

    def get_depth(self, node, dep=0):
        # 递归计算树深度
        global max_depth
        if node:
            dep += 1
        if node.lchild:
            self.get_depth(node.lchild, dep)
        if node.rchild:
            self.get_depth(node.rchild, dep)
        if dep > max_depth:
            max_depth = dep
        return max_depth


"""
dlr、ldr、lrd、lv使用递归实现遍历
_stack使用堆栈实现遍历
_depth层数深度操作
"""

if __name__ == '__main__':
    data = range(10)
    tree = BTree()
    for i in data:
        tree.add_node(i)
    print(tree.get_depth(tree.root))


