using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PointsGUI
{
    // TODO: struct with get methods
    using Cell = System.Byte;
    using Move = System.Int16;
    public class Field
    {
        private readonly Cell[,] field;
        public int Width { get; protected set; }
        public int Height { get; protected set; }
        public readonly List<List<Move>> chains = new List<List<Move>>();
        public Cell this[int i, int j]
        {
            get { return field[i, j]; }
        }

        public Field(int width, int height)
        {
            Width = width;
            Height = height;
            field = new Cell[width, height];
        }
        public bool MakeMove(Move move)
        {
            return false;
        }
        public void Undo()
        {

        }
    }
}
