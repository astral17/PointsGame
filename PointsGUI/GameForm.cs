using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace PointsGUI
{
    public partial class GameForm : Form
    {
        Field field = new Field(20, 20);
        //DllLoader dll = new DllLoader();
        public GameForm()
        {
            InitializeComponent();
            //DllLoader.DllInit(1,2,3);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void gameCanvas_Paint(object sender, PaintEventArgs e)
        {
            //e.Graphics.Clear(Color.White);
            e.Graphics.DrawLine(Pens.Black, 0, 0, 640, 640);
            for (int i = 0; i < 20; i++)
            {
                e.Graphics.DrawString(i.ToString(), DefaultFont, Brushes.Black, 0, i * 32 + 16);
                e.Graphics.DrawString(i.ToString(), DefaultFont, Brushes.Black, i * 32 + 16, 0);
                e.Graphics.DrawLine(Pens.Black, 0, i * 32 + 16, 640, i * 32 + 16);
                e.Graphics.DrawLine(Pens.Black, i * 32 + 16, 0, i * 32 + 16, 640);
            }
        }
    }
}
